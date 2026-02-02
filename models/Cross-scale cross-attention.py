import torch
import torch.nn.functional as F
from timm.layers import DropPath
from typing import List
from torch import nn, Tensor

from model_build import build_norm, build_act


class Mlp(nn.Module):
    def __init__(self, in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, dim,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.v = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.gamma = nn.Parameter(torch.ones(1))  # 可学习的融合权重
        self.linear = nn.Linear(2*dim, dim)

    def forward(self, x, context=None):
        """
        Args:
            x (Tensor):: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(B_, N, 3, C).permute(2, 0, 1, 3)
        q = qkv[0].reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = qkv[1].reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = qkv[2]
        v = self.v(v)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # 通道间注意力而非序列间注意力(C//num_heads, C//num_heads)
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn) # [B_, num_heads, C//num_heads, C//num_heads]

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if context is not None: # 类似交叉注意力
            fusion = (k @ context.transpose(-2, -1)) * torch.sigmoid(self.gamma)
            fusion = self.attn_drop((fusion * self.scale).softmax(dim=-1))
            fusion = (fusion @ v).transpose(1, 2).reshape(B_, N, C)
            x = torch.cat((x, fusion), dim=2)
            x = self.linear(x)
        else:
            x = self.proj(x)
        x = self.proj_drop(x)
        return q, x 


class ChannelTransformerBlock(nn.Module):
    def __init__(self, dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 attn_conv1d=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        if attn_conv1d:
            self.attn_C = ChannelAttention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn_C = ChannelAttention(
                dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, context=None):

        shortcut = x
        x = self.norm1(x)

        # W-MSA/SW-MSA
        q, x = self.attn_C(x, context)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return q, x


class ChannelTransformerLayerCls_fusion_Q(nn.Module):
    def __init__(self, 
                 dim,
                 depth,
                 num_heads,
                 cls,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 attn_conv1d=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.cls = cls

        #修改
        if self.cls:
            self.cls_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, cls, dim)) for _ in range(3)
            ])

        if isinstance(norm_layer, str):
            norm_layer = build_norm(norm_layer)
        if isinstance(act_layer, str):
            act_layer = build_act(act_layer)

        # build blocks(修改)
        self.multi_blocks = nn.ModuleList([
            nn.ModuleList([
                ChannelTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop,
                    drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    attn_conv1d=attn_conv1d
                ) for j in range(depth)
            ]) for _ in range(3)  # 多个独立的blocks
        ])

        self.linear_2 = nn.Linear(136, 267)
        self.linear_1 = nn.Linear(267, 528) 

    #修改x的长度记得同步修改上面的cls_tokens和multi_blocks
    def forward(self, x: List[Tensor]): 
        if self.cls:
            for i in range(len(x)):
                cls_token = self.cls_tokens[i].expand(x[i].shape[0], -1, -1)
                x[i] = torch.cat((cls_token, x[i]), dim=1)

        for blk in self.multi_blocks[2]:
            q2, x[2] = blk(x[2])
            q2 = self.linear_2(q2)
        for blk in self.multi_blocks[1]:
            q1, x[1] = blk(x[1], q2)
            q1 = self.linear_1(q1)
        for blk in self.multi_blocks[0]:
            _, x[0] = blk(x[0], q1)
                 
        return x