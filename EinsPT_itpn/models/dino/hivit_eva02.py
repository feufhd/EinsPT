# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint

from util.misc import NestedTensor
from .utils_eva02 import Block, _cfg, PatchMerge, PatchEmbedMIM, RelativePositionBias, \
    DecoupledRelativePositionBias, RescaleInput
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
# from detectron2.modeling import BACKBONE_REGISTRY, Backbone,ShapeSpec
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.layers.shape_spec import ShapeSpec

# from apex.normalization import FusedLayerNorm

from .rope import *


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


# @BACKBONE_REGISTRY.register()
class HiViTMIM_eva02(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 predict_feature_dim=768,
                 embed_dim=512,
                 mlp_depth1=3,
                 mlp_depth2=3,
                 depth=24,
                 num_heads=8,
                 bridge_mlp_ratio=3.,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.0,
                 init_std=0.02,
                 init_values=None,
                 norm_layer=nn.LayerNorm,
                 attn_head_dim=None,
                 patch_norm=False,
                 grad_ckpt=False,
                 stop_grad_conv1=False,

                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False,
                 use_shared_decoupled_rel_pos_bias=False,
                 rope=False,

                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 xavier_normal_init=False,
                 **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.num_main_blocks = depth
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbedMIM(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
            norm_layer=norm_layer if patch_norm else None, stop_grad_conv1=stop_grad_conv1)
        num_patches = self.patch_embed.num_patches

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim * 4))

        # absolute position embedding
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if use_shared_decoupled_rel_pos_bias:
            assert self.rel_pos_bias is None
            self.rel_pos_bias = DecoupledRelativePositionBias(window_size=self.patch_embed.patch_shape,
                                                              num_heads=num_heads)

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else:
            self.rope = None

        self.subln = subln
        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, mlp_depth1 + mlp_depth2 + depth))

        self.blocks = nn.ModuleList()
        self.blocks.extend([
            Block(
                dim=mlvl_dims['4'], num_heads=0, mlp_ratio=bridge_mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
                depth=mlp_depth1 + mlp_depth2 + depth,
                postnorm=postnorm,
                deepnorm=deepnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=self.rope,
            ) for _ in range(mlp_depth1)]
        )
        self.blocks.append(PatchMerge(mlvl_dims['4'], norm_layer))
        self.blocks.extend([
            Block(
                dim=mlvl_dims['8'], num_heads=0, mlp_ratio=bridge_mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
                depth=mlp_depth1 + mlp_depth2 + depth,
                postnorm=postnorm,
                deepnorm=deepnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=self.rope,
            ) for _ in range(mlp_depth2)]
        )
        self.blocks.append(PatchMerge(mlvl_dims['8'], norm_layer))
        self.blocks.extend([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr), norm_layer=norm_layer,
                init_values=init_values,
                window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
                depth=mlp_depth1 + mlp_depth2 + depth,
                postnorm=postnorm,
                deepnorm=deepnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=self.rope,
            ) for _ in range(depth)]
        )

        self.norm = norm_layer(embed_dim) if not deepnorm else nn.Identity()

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, predict_feature_dim)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)

        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)

        if xavier_normal_init:
            self.apply(self._xavier_normal_init)
            w = self.patch_embed.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        else:  # ori BEiT init
            self.apply(self._init_weights)
            self.fix_init_weight()

        if postnorm:
            self._reinit_respostnorm_ln()

        if deepnorm:
            init_scale = math.pow(8.0 * depth, 0.25)
            for name, p in self.named_parameters():
                if (
                        'mlp.fc' in name
                        or 'mlp.w' in name
                        or 'attn.proj' in name
                        or 'attn.v_proj' in name
                ):
                    print('deepnorm rescale:', name, '/', init_scale)
                    p.data.div_(init_scale)

        if subln:
            init_scale = math.sqrt(math.log(depth * 2))
            for name, p in self.named_parameters():
                if (
                        'mlp.fc' in name
                        or 'mlp.w' in name
                        or 'attn.proj' in name
                        or 'attn.v_proj' in name
                ):
                    print('subln rescale:', name, 'x', init_scale)
                    p.data.mul_(init_scale)

        self.grad_ckpt = grad_ckpt
        self.stop_grad_conv1 = stop_grad_conv1
        self.use_checkpoint = False

    def _reinit_respostnorm_ln(self):
        for blk in self.blocks:
            if blk.norm1 is not None:
                nn.init.constant_(blk.norm1.bias, 0)
                nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if getattr(layer, "merge", False):
                continue
            if layer.attn is not None:
                rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.swiglu or self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _xavier_normal_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def get_num_layers(self):
        return len(self.blocks)

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed
        # npatch = x.shape[1]
        # N = self.absolute_pos_embed.shape[1]
        # if npatch == N and w == h:
        #     return self.absolute_pos_embed
        # patch_pos_embed = self.absolute_pos_embed
        # dim = x.shape[-1]
        # w0 = w // self.patch_size
        # h0 = h // self.patch_size
        # # we add a small number to avoid floating point error in the interpolation
        # # see discussion at https://github.com/facebookresearch/dino/issues/8
        # w0, h0 = w0 + 0.1, h0 + 0.1
        # patch_pos_embed = nn.functional.interpolate(
        #     patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        #     scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
        #     mode='bicubic',
        # )  # 1 256 384->1 384 16 44
        # assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        # patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)  # 1 384 16 44->1 704 384;704=16*44
        # return patch_pos_embed

    def forward_features(self, x, bool_masked_pos=None):
        B, C, H, W = x.shape
        x = self.patch_embed(x, bool_masked_pos, self.mask_token)

        for blk in self.blocks[:-self.num_main_blocks]:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)

        x = x[..., 0, 0, :]

        if self.pos_embed is not None:
            ###add
            x = x + self.interpolate_pos_encoding(x, H, W)
            ###
            # x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        if self.grad_ckpt:
            for blk in self.blocks[-self.num_main_blocks:]:
                x = torch.utils.checkpoint.checkpoint(blk, x, rel_pos_bias) if self.use_checkpoint else blk(x,
                                                                                                            rel_pos_bias)
        else:
            for blk in self.blocks[-self.num_main_blocks:]:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)

        return x

    def forward(self, x, bool_masked_pos=None):
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        x = self.lm_head(x[bool_masked_pos])
        return x


# @BACKBONE_REGISTRY.register()
class HiViT_eva02(HiViTMIM_eva02, Backbone):
    def __init__(self, pretrain_img_size=256,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=512,
                 mlp_depth1=3,
                 mlp_depth2=3,
                 depths=24,
                 num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=True,
                 rpe=False,  # true
                 patch_norm=True,
                 grad_ckpt=False,
                 init_values=0.1,

                 postnorm=False,
                 deepnorm=False,
                 subln=True,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=True,
                 xavier_normal_init=True,
                 stop_grad_conv1=True,
                 rope=False,
                 ):

        super().__init__(
            img_size=pretrain_img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            # predict_feature_dim=pretrain_img_size,
            predict_feature_dim=512,
            embed_dim=embed_dim,
            mlp_depth1=mlp_depth1,
            mlp_depth2=mlp_depth2,
            depth=depths,
            num_heads=num_heads,
            bridge_mlp_ratio=3. * 2 / 3,
            mlp_ratio=4. * 2 / 3,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            init_std=0.02,
            patch_norm=patch_norm,
            init_values=init_values,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),  # partial(FusedLayerNorm, eps=1e-6),
            grad_ckpt=grad_ckpt,
            stop_grad_conv1=stop_grad_conv1,

            use_abs_pos_emb=ape,
            use_rel_pos_bias=rpe,
            use_shared_rel_pos_bias=False,
            use_shared_decoupled_rel_pos_bias=False,
            rope=rope,

            postnorm=postnorm,
            deepnorm=deepnorm,
            subln=subln,
            xattn=xattn,
            swiglu=swiglu,
            naiveswiglu=naiveswiglu,
            xavier_normal_init=xavier_normal_init,
        )

        self.num_features = [embed_dim // 4, embed_dim // 2, embed_dim, embed_dim * 2]
        ##
        self._out_features = ['s1', 's2', 's3']

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            # "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            # "res5": self.num_features[3],
        }
        self.rescale = RescaleInput(padding='corner', kernel_size=patch_size, stride=patch_size, dilation=1)

    def forward_features(self, x, bool_masked_pos=None):
        x = self.rescale(x)

        B, C, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x = self.patch_embed(x, bool_masked_pos, self.mask_token)

        outs = []
        for blk in self.blocks[:-self.num_main_blocks]:
            if isinstance(blk, PatchMerge):
                outs.append(x)
            x = checkpoint.checkpoint(blk, x) if self.grad_ckpt else blk(x)

        x = x[..., 0, 0, :]

        if self.pos_embed is not None:
            ##add
            x = x + self.interpolate_pos_encoding(x, H, W)
            ##
            # x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for blk in self.blocks[-self.num_main_blocks:]:
            x = checkpoint.checkpoint(blk, x, rel_pos_bias) if self.grad_ckpt else blk(x, rel_pos_bias)

        x = self.norm(x)

        outs.append(x)

        outs[0] = outs[0].transpose(1, 2).reshape(B, -1, Hp * 4, Wp * 4)
        outs[1] = outs[1].transpose(1, 2).reshape(B, -1, Hp * 2, Wp * 2)
        outs[2] = outs[2].transpose(1, 2).reshape(B, -1, Hp, Wp)

        # outs.append(F.avg_pool2d(outs[-1], kernel_size=2, stride=2))

        return outs

    def forward(self, x_all, bool_masked_pos=None):
        x = x_all.tensors
        outs = self.forward_features(x, bool_masked_pos=bool_masked_pos)

        # outputs = {}
        # for i, k in enumerate(self._out_features):
        #     outputs[k] = x[i]
        # return outputs

        outs_dict = {}
        for idx, out_i in enumerate(self._out_features):
            m = x_all.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=outs[idx].shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(outs[idx], mask)

        return outs_dict

    def output_shape(self):

        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32


def build_hivit_eva02_transformer(modelname, **kw):
    assert modelname in ['hivit_eva02']

    model_para_dict = {
        'hivit_eva02': dict(
            pretrain_img_size=256,
            patch_size=16,
            in_chans=3,
            embed_dim=512,
            mlp_depth1=3,
            mlp_depth2=3,
            depths=24,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            ape=True,
            rpe=False,  # true
            patch_norm=True,
            grad_ckpt=False,
            init_values=0.1,

            postnorm=False,
            deepnorm=False,
            subln=True,
            xattn=False,
            swiglu=False,
            naiveswiglu=True,
            xavier_normal_init=True,
            stop_grad_conv1=True,
            rope=False,
        ),

    }
    kw_cgf = model_para_dict[modelname]
    kw_cgf.update(kw)
    model = HiViT_eva02(**kw_cgf)
    return model
