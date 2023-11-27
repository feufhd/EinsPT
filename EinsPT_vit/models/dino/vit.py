# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# --------------------------------------------------------
# modified from https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/mmdet/models/backbones/swin_transformer.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from util.misc import NestedTensor

from functools import partial
import timm.models.vision_transformer
from timm.models.registry import register_model
import math


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=True, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        
        self.pos_embed = nn.Parameter(torch.randn(1, 197, 768))
            
    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        cls_pos_embed = self.pos_embed[:, 0, :].reshape(1, 1, x.shape[-1])
        patch_pos_embed = self.pos_embed[:, 1:, :].reshape(1, -1, x.shape[-1])
        dim = x.shape[-1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w + 0.1, h + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        patch_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
        return patch_pos_embed
            
    def forward_features(self, x, bool_masked_pos=None):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        Hp, Wp = H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1]
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        if bool_masked_pos is not None:
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_token
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_embed is not None:
            x = x + self.interpolate_pos_encoding(x, Hp, Wp)
                
        x = self.pos_drop(x)
        
        outcome = []
        
        #rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        idx = 0
        for blk in self.blocks:
            x = blk(x)
            if idx in [4, 7, 11]:
                outcome.append(self.fc_norm(x[:, 1:, :]).permute(0, 2, 1).reshape(B, -1, Hp, Wp))
            idx += 1
            
        outcome.append(self.fc_norm(x[:, 1:, :]).permute(0, 2, 1).reshape(B, -1, Hp, Wp))
            
        return outcome
    
    def forward(self, x_all, bool_masked_pos=None, aux_shapes=None):
        x = x_all.tensors
        features = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        
        # print(' >>>>>>>>>>>>>>>>>>>>>> aux_shapes :', aux_shapes)
        features[0] = nn.functional.interpolate(features[0], scale_factor=(4, 4), mode='bicubic')
        features[1] = nn.functional.interpolate(features[1], scale_factor=(2, 2), mode='bicubic')
        features[3] = nn.functional.interpolate(features[3], scale_factor=(0.5, 0.5), mode='bicubic')
        
        outs = {}
        for i in range(4):
            m = x_all.mask
            mask = F.interpolate(m[None].float(), size=aux_shapes[i][-2:]).to(torch.bool)[0]
            # outs["p{}".format(i+2)] = NestedTensor(features[i],mask)
            outs[i] = NestedTensor(features[i], mask)
        return outs
            

@register_model
def vit_base_patch16(modelname, pretrained=False, **kwargs):
    assert modelname in ['ViT_base']
    model_para_dict = {
        'ViT_base': dict(
            img_size=256,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        ), }
    kw_cgf = model_para_dict[modelname]
    kw_cgf.update(kwargs)
    model = VisionTransformer(**kw_cgf)
    #model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model