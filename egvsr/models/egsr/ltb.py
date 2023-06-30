import math

import torch
import torch.nn as nn
from absl.logging import info

from egvsr.models.egsr.patch_options import extract_patches, restore_patches
from egvsr.models.utils import pair


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EffAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.reduce = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim // 2, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        q_all = torch.split(q, math.ceil(N // 4), dim=-2)
        k_all = torch.split(k, math.ceil(N // 4), dim=-2)
        v_all = torch.split(v, math.ceil(N // 4), dim=-2)

        output = []
        for q, k, v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            trans_x = (attn @ v).transpose(1, 2)
            output.append(trans_x)
        x = torch.cat(output, dim=1)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        return x


class EfficientTransformer(nn.Module):
    def __init__(self, dim, drop):
        super(EfficientTransformer, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = EffAttention(
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim // 4,
            act_layer=nn.ReLU,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MLABlock(nn.Module):
    def __init__(self, dim, drop=0.0, depth=1):
        super(MLABlock, self).__init__()
        ets = []
        for i in range(depth):
            ets.append(EfficientTransformer(dim, drop))
        self.ets = nn.Sequential(*ets)

    def forward(self, x):
        x = self.ets(x)
        return x


class MLABBlock4D(nn.Module):
    def __init__(self, feature_size, patch_size, dim, drop=0.0, depth=1):
        super(MLABBlock4D, self).__init__()

        self.feature_size = feature_size
        self.patch_size = patch_size
        self.mlab = MLABlock(dim, drop, depth)

        info(f"Init MLABBlock4D")
        info(f"  --feature_size: {feature_size}")
        info(f"  --patch_size  : {patch_size}")
        info(f"  --dim         : {dim}")
        info(f"  --drop        : {drop}")
        info(f"  --depth       : {depth}")

    def forward(self, x):
        x = extract_patches(x, self.patch_size, [1, 1], [1, 1])
        x = x.permute(0, 2, 1)
        y = self.mlab(x)
        y = y.permute(0, 2, 1)
        y = restore_patches(
            y,
            self.feature_size,
            self.patch_size[0],
            1,
            self.patch_size[0] // 2,
        )
        return y
