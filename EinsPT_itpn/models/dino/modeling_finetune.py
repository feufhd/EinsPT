# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Yunjie Tian
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from functools import partial

import math
import torch
import torch.nn as nn
from timm.models.registry import register_model
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple, drop_path, trunc_normal_

from torch import Tensor, Size
from typing import Union, List

# import xformers.ops as xops
# from apex.normalization import FusedLayerNorm

from .rope import *
from util.misc import NestedTensor
from .utils_eva02 import RescaleInput


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 21841, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


_shape_t = Union[int, List[int], Size]


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.ffn_ln(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else None

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.ffn_ln is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.ffn_ln(x)
            x = x.permute(0, 3, 1, 2)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class ConvSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Conv2d(in_features, hidden_features, 1)
        self.w2 = nn.Conv2d(in_features, hidden_features, 1)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Conv2d(hidden_features, out_features, 1)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = self.w1(x).flatten(2).transpose(1, 2)
        x2 = self.w2(x).flatten(2).transpose(1, 2)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden).transpose(1, 2).view(B, C, H, W)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=None,
            attn_head_dim=None, use_decoupled_rel_pos_bias=False, deepnorm=False, subln=False, norm_layer=nn.LayerNorm,
            xattn=False, rope=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.deepnorm = deepnorm
        self.subln = subln
        if self.deepnorm or self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rel_pos_bias = None
        self.qk_float = True

        self.window_size = None
        self.relative_position_bias_table = None

        if window_size:
            if use_decoupled_rel_pos_bias:
                self.rel_pos_bias = DecoupledRelativePositionBias(window_size=window_size, num_heads=num_heads)
            else:
                self.window_size = window_size
                self.num_relative_distance = (2 * window_size[0] - 1) * (
                        2 * window_size[1] - 1) + 3  # (2*14-1) * (2*14-1) + 3
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
                # cls to token & token 2 cls & cls to cls

                # get pair-wise relative position index for each token inside the window
                coords_h = torch.arange(window_size[0])
                coords_w = torch.arange(window_size[1])
                coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
                coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
                relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
                relative_coords[:, :, 1] += window_size[1] - 1
                relative_coords[:, :, 0] *= 2 * window_size[1] - 1
                relative_position_index = \
                    torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
                relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
                relative_position_index[0, 0:] = self.num_relative_distance - 3
                relative_position_index[0:, 0] = self.num_relative_distance - 2
                relative_position_index[0, 0] = self.num_relative_distance - 1

                self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.xattn = xattn
        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape

        if self.deepnorm or self.subln:
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, N, C
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        else:
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope is not None:
            # q_t = q[:, :, 1:, :]
            # ro_q_t = self.rope(q_t)
            # q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

            # k_t = k[:, :, 1:, :]
            # ro_k_t = self.rope(k_t)
            # k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)

            q = self.rope(q).type_as(v)

            k = self.rope(k).type_as(v)

        if self.xattn:
            q = q.permute(0, 2, 1, 3)  # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            x = xops.memory_efficient_attention(q, k, v)
            x = x.reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            q = q * self.scale
            if self.qk_float:
                attn = (q.float() @ k.float().transpose(-2, -1))
            else:
                attn = (q @ k.transpose(-2, -1))

            if self.relative_position_bias_table is not None:
                relative_position_bias = \
                    self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

            if self.rel_pos_bias is not None:
                attn = attn + self.rel_pos_bias().type_as(attn)

            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.type_as(attn)
            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = attn.softmax(dim=-1).type_as(x)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, use_decoupled_rel_pos_bias=False,
                 depth=None,
                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=None,
                 ):
        super().__init__()

        with_attn = num_heads > 0

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size,
            use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias, attn_head_dim=attn_head_dim,
            deepnorm=deepnorm,
            subln=subln,
            norm_layer=norm_layer,
            xattn=xattn,
            rope=rope,
        ) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        if swiglu:
            self.mlp = xops.SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim
            )  # hidden_features: 2/3
        elif naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),
                                        requires_grad=True) if self.attn is not None else None
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm:
            self.alpha = math.pow(2.0 * depth, 0.25)

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_2 is None:
            if self.postnorm:
                if self.attn is not None:
                    x = x + self.drop_path(
                        self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            elif self.deepnorm:
                if self.attn is not None:
                    residual = x
                    x = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                    x = self.drop_path(x)
                    x = residual * self.alpha + x
                    x = self.norm1(x)

                residual = x
                x = self.mlp(x)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm2(x)
            else:
                if self.attn is not None:
                    x = x + self.drop_path(
                        self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                if self.attn is not None:
                    x = x + self.drop_path(
                        self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                if self.attn is not None:
                    x = x + self.drop_path(
                        self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class ConvMlpBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, use_decoupled_rel_pos_bias=False,
                 depth=None,
                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=None,
                 convmlp=False,
                 ):
        super().__init__()

        self.attn = None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        if swiglu:
            self.mlp = xops.SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim
            )  # hidden_features: 2/3
        elif naiveswiglu:
            self.mlp = ConvSwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = ConvMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),
                                        requires_grad=True) if self.attn is not None else None
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm:
            self.alpha = math.pow(2.0 * depth, 0.25)

        self.postnorm = postnorm

    def forward(self, x):
        # print('---------------------------------', x.shape)
        if self.gamma_2 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            elif self.deepnorm:
                residual = x
                x = self.mlp(x)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm2(x)
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        else:
            if self.postnorm:
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                m = self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
                # print('------------------------- m:', m.shape)
                x = x + self.drop_path(self.gamma_2 * m)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=128, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = self.patch_shape = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1,
            patches_resolution[0], self.inner_patches,
            patches_resolution[1], self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=128, norm_layer=None,
                 stop_grad_conv1=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.stop_grad_conv1 = stop_grad_conv1
        self.inner_patches = inner_patches
        self.patches_resolution = self.patch_shape = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, bool_masked_pos=None, mask_token=None):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.stop_grad_conv1:
            x = x.detach() * 0.9 + x * 0.1

        if bool_masked_pos is not None:
            x = torch.nn.functional.unfold(x, kernel_size=4, stride=4, padding=0).transpose(1, 2)

            seq_len = x.shape[1]
            mask_token = mask_token.expand(B, seq_len, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

            x = torch.nn.functional.fold(x.transpose(1, 2), output_size=(H // 4, W // 4), kernel_size=4, padding=0,
                                         stride=4)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)
        self.mlp = None

    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class ConvPatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.reduction = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2, padding=0)
        self.mlp = None

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.reduction(x)
        return x


class PatchSplit(nn.Module):
    def __init__(self, dim, fpn_dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.reduction = nn.Linear(dim, fpn_dim * 4, bias=False)
        self.fpn_dim = fpn_dim

    def forward(self, x):
        B, N, H, W, C = x.shape
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(
            B, N, H, W, 2, 2, self.fpn_dim
        ).permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, N, 2 * H, 2 * W, self.fpn_dim
        )
        return x


class ConvPatchSplit(nn.Module):
    def __init__(self, dim, fpn_dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.reduction = torch.nn.ConvTranspose2d(dim, fpn_dim, kernel_size=2, stride=2, padding=0)
        self.fpn_dim = fpn_dim

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.reduction(x)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


def _mask_1d_rel_pos_index(seq_len):
    index = torch.arange(seq_len)
    return index.view(1, seq_len) - index.view(seq_len, 1) + seq_len - 1


def _add_cls_to_index_matrix(index, num_tokens, offset):
    index = index.contiguous().view(num_tokens, num_tokens)
    new_index = torch.zeros(size=(num_tokens + 1, num_tokens + 1), dtype=index.dtype)
    new_index[1:, 1:] = index
    new_index[0, 0:] = offset
    new_index[0:, 0] = offset + 1
    new_index[0, 0] = offset + 2
    return new_index


class DecoupledRelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] + 2, 2 * window_size[1] + 2)

        num_tokens = window_size[0] * window_size[1]

        self.relative_position_bias_for_high = nn.Parameter(torch.zeros(self.num_relative_distance[0], num_heads))
        self.relative_position_bias_for_width = nn.Parameter(torch.zeros(self.num_relative_distance[1], num_heads))
        # cls to token & token 2 cls & cls to cls

        h_index = _mask_1d_rel_pos_index(window_size[0]).view(
            window_size[0], 1, window_size[0], 1).expand(-1, window_size[1], -1, window_size[1])
        h_index = _add_cls_to_index_matrix(h_index, num_tokens, 2 * window_size[0] - 1)
        self.register_buffer("relative_position_high_index", h_index)

        w_index = _mask_1d_rel_pos_index(window_size[1]).view(
            1, window_size[1], 1, window_size[1]).expand(window_size[0], -1, window_size[0], -1)
        w_index = _add_cls_to_index_matrix(w_index, num_tokens, 2 * window_size[1] - 1)

        self.register_buffer("relative_position_width_index", w_index)

    def forward(self):
        relative_position_bias = \
            F.embedding(input=self.relative_position_high_index, weight=self.relative_position_bias_for_high) + \
            F.embedding(input=self.relative_position_width_index, weight=self.relative_position_bias_for_width)
        return relative_position_bias.permute(2, 0, 1).contiguous()


class iTPN(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24,
                 num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4., fpn_dim=256, fpn_depth=2, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, init_values=0.1, attn_head_dim=None,
                 norm_layer=nn.LayerNorm, patch_norm=False, use_checkpoint=True, num_classes=1000,
                 use_mean_pooling=False,
                 init_std=0.02,
                 init_scale=0.01,
                 cls_token=False,
                 grad_ckpt=True,
                 stop_grad_conv1=False,
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False,
                 use_shared_decoupled_rel_pos_bias=False,
                 rope=False,
                 convmlp=False,

                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 xavier_normal_init=False,
                 # out_num=4,
                 **kwargs):
        super().__init__()
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.grad_ckpt = grad_ckpt
        self.num_main_blocks = depth
        self.depth_stage1 = depth_stage1
        self.depth_stage2 = depth_stage2
        self.depth = depth
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.convmlp = convmlp
        self.stop_grad_conv1 = stop_grad_conv1

        # mlvl_dims={'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        # if out_num==3:
        #     mlvl_dims=mlvl_dims
        # elif out_num==4:
        #     mlvl_dims .update({'32':embed_dim*2})
        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim, '32': embed_dim * 2}

        ###add
        self.mlvl_dims = mlvl_dims
        self.downsample = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=2, stride=2)
        # split image into non-overlapping patches
        if convmlp:
            self.patch_embed = ConvPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
                stop_grad_conv1=stop_grad_conv1,
                norm_layer=norm_layer if patch_norm else None)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
                norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches

        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        if use_abs_pos_emb:
            if cls_token:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            else:
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
                pt_seq_len=hw_seq_len,
            )
        else:
            self.rope = None

        self.subln = subln
        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        self.num_features = [embed_dim // 4, embed_dim // 2, embed_dim, embed_dim * 2]
        ##
        # self._out_features = ['s1', 's2', 's3','s4']

        # self._out_feature_strides = {
        #     "res2": 4,
        #     "res3": 8,
        #     "res4": 16,
        #     # "res5": 32,
        # }
        # self._out_feature_channels = {
        #     "res2": self.num_features[0],
        #     "res3": self.num_features[1],
        #     "res4": self.num_features[2],
        #     # "res5": self.num_features[3],
        # }
        self.rescale = RescaleInput(padding='corner', kernel_size=patch_size, stride=patch_size, dilation=1)

        self.build_blocks(
            depths=[depth_stage1, depth_stage2, depth],
            dims=mlvl_dims,
            num_heads=num_heads,
            bridge_mlp_ratio=bridge_mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            attn_head_dim=attn_head_dim,
            postnorm=postnorm,
            deepnorm=deepnorm,
            subln=subln,
            xattn=xattn,
            swiglu=swiglu,
            naiveswiglu=naiveswiglu,
            rope=self.rope,
            convmlp=convmlp,
        )

        ## FPN PART modules are not needed for classification, we delete them here ##

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        # if isinstance(self.head, nn.Linear):
        #     trunc_normal_(self.head.weight, std=.02)

        self.apply(self._init_weights)
        # self.fix_init_weight()

        # if isinstance(self.head, nn.Linear):
        # self.head.weight.data.mul_(init_scale)
        # self.head.bias.data.mul_(init_scale)

    def build_blocks(self,
                     depths=[3, 3, 24],
                     dims={'4': 128 // 4, '8': 256, '16': 512},
                     num_heads=8,
                     bridge_mlp_ratio=3.,
                     qkv_bias=True,
                     qk_scale=None,
                     window_size=None,
                     drop=0.,
                     attn_drop=0.,
                     drop_path_rate=0.,
                     norm_layer=nn.LayerNorm,
                     init_values=0.,
                     attn_head_dim=None,
                     postnorm=False,
                     deepnorm=False,
                     subln=False,
                     xattn=False,
                     swiglu=False,
                     naiveswiglu=False,
                     rope=False,
                     convmlp=False,
                     ):
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, depths[0] + depths[1] + depths[2]))

        self.blocks = nn.ModuleList()

        if convmlp:
            self.blocks.extend([
                ConvMlpBlock(
                    dim=dims['4'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=0.,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    xattn=xattn,
                    swiglu=False,
                    naiveswiglu=False,
                    rope=rope
                ) for _ in range(depths[0])
            ])
            self.blocks.append(ConvPatchMerge(dims['4'], norm_layer))
            self.blocks.extend([
                ConvMlpBlock(
                    dim=dims['8'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=0.,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    xattn=xattn,
                    swiglu=False,
                    naiveswiglu=False,
                    rope=rope
                ) for _ in range(depths[1])
            ])
            self.blocks.append(ConvPatchMerge(dims['8'], norm_layer))
        else:
            self.blocks.extend([
                Block(
                    dim=dims['4'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    xattn=xattn,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                    rope=self.rope
                ) for _ in range(depths[0])
            ])
            self.blocks.append(PatchMerge(dims['4'], norm_layer))
            self.blocks.extend([
                Block(
                    dim=dims['8'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    xattn=xattn,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                    rope=self.rope
                ) for _ in range(depths[1])
            ])
            self.blocks.append(PatchMerge(dims['8'], norm_layer))

        ######### stage 3 ########
        self.blocks.extend([
            Block(
                dim=dims['16'],
                num_heads=num_heads,
                mlp_ratio=bridge_mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=next(dpr),
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=window_size,
                attn_head_dim=attn_head_dim,
                depth=depths[-1],
                postnorm=postnorm,
                deepnorm=deepnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=rope
            ) for _ in range(depths[2])
        ])

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if layer.attn is not None:
                rescale(layer.attn.proj.weight.data, layer_id + 1)
            if layer.mlp is not None:
                if self.swiglu or self.naiveswiglu:
                    rescale(layer.mlp.w3.weight.data, layer_id + 1)
                else:
                    rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cls_token is not None:
            return {'pos_embed', 'cls_token'}
        return {'pos_embed'}

    def get_classifer(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        # self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

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

    def forward_features(self, x):
        outs = []
        x = self.rescale(x)

        B, C, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x = self.patch_embed(x)

        if not self.convmlp and self.stop_grad_conv1:
            x = x.detach() * 0.9 + x * 0.1

        for blk in self.blocks[:-self.num_main_blocks]:
            if isinstance(blk, PatchMerge):
                out = x.flatten(2).transpose(1, 2)
                outs.append(out)
            elif isinstance(blk, ConvPatchMerge):
                outs.append(x)
            x = checkpoint.checkpoint(blk, x) if self.grad_ckpt else blk(x)

        x = x.flatten(2).transpose(1, 2)
        # x = x[..., 0, 0, :]

        # if self.cls_token is not None:
        #     cls_tokens = self.cls_token.expand(B, -1, -1)
        #     x = torch.cat([cls_tokens, x], dim=1)
        if self.pos_embed is not None:
            # x = x + self.pos_embed
            x = x + self.interpolate_pos_encoding(x, H, W)
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks[-self.num_main_blocks:]:
            x = checkpoint.checkpoint(blk, x, rel_pos_bias) if self.grad_ckpt else blk(x, rel_pos_bias)

        x = self.norm(x)
        outs.append(x)

        outs[0] = outs[0].reshape(B, -1, Hp * 4, Wp * 4)
        outs[1] = outs[1].reshape(B, -1, Hp * 2, Wp * 2)
        outs[2] = outs[2].transpose(1, 2).reshape(B, -1, Hp, Wp)

        if len(self.mlvl_dims) == 4:
            outs.append(self.downsample(outs[2]))

        return outs
        # if self.fc_norm is not None:
        #     if self.cls_token is not None:
        #         t = x[:, 1:, :]
        #     return self.fc_norm(x.mean(1))
        # else:
        #     # return x[:, 0]
        #     return x.mean(1)

    def forward(self, x_all):
        x = x_all.tensors
        outs = self.forward_features(x)
        # x = self.head(x)
        outs_dict = {}

        # mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim, '32': embed_dim * 2}
        for idx, out_i in enumerate(self.mlvl_dims):
            m = x_all.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=outs[idx].shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(outs[idx], mask)

        return outs_dict


@register_model
def itpn_base_3324_patch16_224(pretrained=False, **kwargs):
    model = iTPN(
        patch_size=16, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def itpn_base_2240_patch16_224(pretrained=False, **kwargs):
    model = iTPN(
        patch_size=16, embed_dim=768, depth_stage1=2, depth_stage2=2, depth=40, num_heads=12, bridge_mlp_ratio=3.,
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224(pretrained=False, **kwargs):
    model = iTPN(
        patch_size=16, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
        mlp_ratio=4 * 2 / 3, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        xattn=False,
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        xavier_normal_init=True,
        rope=False,
        num_classes=21841,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


def build_itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224_transformer(modelname, pretrained=False,
                                                                                  **kwargs):
    assert modelname in ['iTPN_base']

    model_para_dict = {
        'iTPN_base': dict(
            # pretrain_img_size=256,
            patch_size=16,
            # in_chans=3,
            embed_dim=512,
            depth_stage1=3, depth_stage2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
            mlp_ratio=4 * 2 / 3, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            xattn=False,
            convmlp=True,
            naiveswiglu=True,
            subln=True,
            xavier_normal_init=True,
            rope=False,
            num_classes=21841,
            grad_ckpt=True,
            use_checkpoint=False,
            **kwargs,
        ),

    }
    kw_cgf = model_para_dict[modelname]
    kw_cgf.update(kwargs)
    model = iTPN(**kw_cgf)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
