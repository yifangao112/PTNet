from typing import Sequence, Tuple, Union
from copy import deepcopy

from torch import nn
import torch
import numpy as np
import torch.nn.functional
import torch.nn.functional as F
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from network_architecture.custom.blocks.dynunet_block import UnetResBlock, UnetResSEBlock, get_conv_layer
from network_architecture.neural_network import SegmentationNetwork


def multiply(pool_op_kernel_sizes, axis, i_layer, num_layers, reverse=False):
    result = 1
    if reverse:
        i_layer = num_layers - i_layer - 1
    for idx in range(i_layer + 1):
        result *= pool_op_kernel_sizes[idx][axis]
    return result


class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
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


def window_partition(x, window_size):
    B, S, H, W, C = x.shape
    x = x.view(B, S // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, S, H, W):
    B = int(windows.shape[0] / (S * H * W / window_size / window_size / window_size))
    x = windows.view(B, S // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, 
                 qk_scale=None, attn_drop=0., proj_drop=0., isfusion=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.isfusion = isfusion

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads)) 

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)

        if self.isfusion:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2=None, mask=None, pos_embed=None):
        B_, N, C = x1.shape

        if self.isfusion:
            q = self.q(x1)
            q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            q = q * self.scale

            kv = self.kv(x2)
            kv = kv.reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
            k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        else:
            qkv = self.qkv(x1)
            qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            q = q * self.scale
        
        attn = (q @ k.transpose(-2, -1).contiguous())
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        output = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()

        if pos_embed is not None:
            output = output + pos_embed
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, isfusion=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.isfusion = isfusion
   
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if self.isfusion:
            self.normx1 = norm_layer(dim)
            self.normx2 = norm_layer(dim)
        else:
            self.normx1 = norm_layer(dim)
        
        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, isfusion=isfusion)
       
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x1, x2=None, mask_matrix=None):
        B, L, C = x1.shape
        S, H, W = self.input_resolution
   
        assert L == S * H * W, "input feature has wrong size"
        
        shortcut = x1

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size

        if self.isfusion:
            x1 = self.normx1(x1)
            x1 = x1.view(B, S, H, W, C)
            x1 = F.pad(x1, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))

            x2 = self.normx2(x2)
            x2 = x2.view(B, S, H, W, C)
            x2 = F.pad(x2, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        else:
            x1 = self.normx1(x1)
            x1 = x1.view(B, S, H, W, C)
            x1 = F.pad(x1, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g)) 

        _, Sp, Hp, Wp, _ = x1.shape

        # cyclic shift
        if self.isfusion:
            if self.shift_size > 0:
                shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, 
                                                    -self.shift_size, -self.shift_size), dims=(1, 2, 3))
                attn_mask = mask_matrix
            else:
                shifted_x1 = x1
                attn_mask = None
            # partition windows
            x1_windows = window_partition(shifted_x1, self.window_size)  # nW*B, window_size, window_size, C
            x1_windows = x1_windows.view(-1, self.window_size * self.window_size * self.window_size, C)

            # x2
            if self.shift_size > 0:
                shifted_x2 = torch.roll(x2, shifts=(-self.shift_size, 
                                                    -self.shift_size, -self.shift_size), dims=(1, 2, 3))
                attn_mask = mask_matrix
            else:
                shifted_x2 = x2
                attn_mask = None
            # partition windows
            x2_windows = window_partition(shifted_x2, self.window_size)  # nW*B, window_size, window_size, C
            x2_windows = x2_windows.view(-1, self.window_size * self.window_size * self.window_size, C)

            # W-MSA/SW-MSA
            attn_windows = self.attn(x1_windows, x2_windows, mask=attn_mask, pos_embed=None)

        else:
            if self.shift_size > 0:
                shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, 
                                                    -self.shift_size, -self.shift_size), dims=(1, 2, 3))
                attn_mask = mask_matrix
            else:
                shifted_x1 = x1
                attn_mask = None
            # partition windows
            x1_windows = window_partition(shifted_x1, self.window_size)  # nW*B, window_size, window_size, C
            x1_windows = x1_windows.view(-1, self.window_size * self.window_size * self.window_size, C)
            # W-MSA/SW-MSA
            attn_windows = self.attn(x1_windows, mask=attn_mask, pos_embed=None)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x1 = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp) 

        # reverse cyclic shift
        if self.shift_size > 0:
            x1 = torch.roll(shifted_x1, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x1 = shifted_x1

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x1 = x1[:, :S, :H, :W, :].contiguous()

        x1 = x1.view(B, S * H * W, C)

        # FFN
        x1 = shortcut + self.drop_path(x1)
        x1 = x1 + self.drop_path(self.mlp(self.norm2(x1)))

        return x1


class MultimodalFusionBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
   
        self.fusion_block1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, 
                 num_heads=num_heads, window_size=window_size, shift_size=0,
                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                 drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                 act_layer=act_layer, norm_layer=norm_layer, isfusion=True)

        self.fusion_block2 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, 
                 num_heads=num_heads, window_size=window_size, shift_size=0,
                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                 drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                 act_layer=act_layer, norm_layer=norm_layer, isfusion=True)

        self.fusion_block3 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, 
                 num_heads=num_heads, window_size=window_size, shift_size=0,
                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                 drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                 act_layer=act_layer, norm_layer=norm_layer, isfusion=True)
        
    def forward(self, x1, x2, x3, mask_matrix):
        x1_out1 = self.fusion_block1(x1, x2, mask_matrix)
        x1_out2 = self.fusion_block2(x1, x3, mask_matrix)
        output = self.fusion_block3(x1_out1, x1_out2, mask_matrix)
        return output


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, stride=(2, 2, 2)):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv3d(dim, dim*2, kernel_size=3, stride=stride, padding=1)
        self.norm = norm_layer(dim)

    def forward(self, x, S, H, W):
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        x = x.view(B, S, H, W, C)
        x = F.gelu(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, 2*C)
        return x


class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, stride=(2, 2, 2)):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.stride = stride
        self.up = nn.ConvTranspose3d(in_channels=dim, 
                                     out_channels=dim//2, 
                                     kernel_size=stride, 
                                     stride=stride)

    def forward(self, x, S, H, W):
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        x = x.view(B, S, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.up(x)
        x = ContiguousGrad.apply(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C//2)
        return x


class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 stride=(2, 2, 2)
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.stride = stride
        # build blocks
        
        self.blocks = nn.ModuleList([
            MultimodalFusionBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer
            ),
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[1] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer,
                isfusion=False)]
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, stride=self.stride)
        else:
            self.downsample = None

    def forward(self, x1, x2, x3, S, H, W):
        # calculate attention mask for SW-MSA
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x1.device)  # 1 Hp Wp 1
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size)  
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for i, blk in enumerate(self.blocks):
            if i == 0:
                x1 = blk(x1, x2, x3, attn_mask)
            else:
                x1 = blk(x1, attn_mask)
            # print('blk', x.shape)

        if self.downsample is not None:
            # print('S, H, W', S, H, W)
            x_down = self.downsample(x1, S, H, W)
            # Ws, Wh, Ww = (S + 1) // self.stride[0], (H + 1) // self.stride[1], (W + 1) // self.stride[2]
            Ws, Wh, Ww = S // self.stride[0], H // self.stride[1], W // self.stride[2]
            return x_down, Ws, Wh, Ww
        else:
            return x1, S, H, W

        
class project(nn.Module):
    def __init__(self,in_dim,out_dim,stride,padding,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.Conv3d(in_dim,out_dim,kernel_size=3,stride=stride,padding=padding)
        self.conv2=nn.Conv3d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last  
        if not last:
            self.norm2=norm(out_dim)
            
    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        
        x = self.conv2(x)
        if not self.last:
            x=self.activate(x)
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        stride1=[patch_size[0], patch_size[1]//2, patch_size[2]//2]
        stride2=[patch_size[0]//2, patch_size[1]//2, patch_size[2]//2]

        self.proj1 = project(in_chans, embed_dim//2, stride1, 1, nn.GELU, nn.LayerNorm, False)
        self.proj2 = project(embed_dim//2, embed_dim, stride2, 1, nn.GELU, nn.LayerNorm, True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, S, H, W = x.size()
        if W % self.patch_size[2] != 0 or H % self.patch_size[1] != 0 or S % self.patch_size[0] != 0:
            raise
        x = self.proj1(x)
        x = self.proj2(x)
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)
        return x

        
class MultimodalSwinT(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=2,
                 in_chans=1,
                 embed_dim=24,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 pool_op_kernel_sizes=None,
                 embed_share_weight=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.in_chans = in_chans
        self.embed_share_weight = embed_share_weight

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # pool
        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = [(2, 2, 2)] * self.num_layers
            pool_op_kernel_sizes.insert(0, [2, 4, 4])

        # split image into non-overlapping patches
        if self.embed_share_weight:
            layer = PatchEmbed(patch_size=patch_size, in_chans=1, embed_dim=embed_dim,
                            norm_layer=norm_layer if self.patch_norm else None)
            layer_name = f'patch_embed'
            self.add_module(layer_name, layer)
        else:
            for i_modality in range(in_chans):
                layer = PatchEmbed(patch_size=patch_size, in_chans=1, embed_dim=embed_dim,
                                norm_layer=norm_layer if self.patch_norm else None)
                layer_name = f'patch_embed{i_modality+1}'
                self.add_module(layer_name, layer)

        for i_modality in range(in_chans):
            layer = nn.Dropout(p=drop_rate)
            layer_name = f'pos_drop{i_modality+1}'
            self.add_module(layer_name, layer)
   
        # build layers
        for i_modality in range(in_chans):
            for i_layer in range(self.num_layers):
                layer_name = f'layer{i_modality+1}_{i_layer}'
                resolution_x = multiply(pool_op_kernel_sizes, 0, i_layer, self.num_layers)
                resolution_y = multiply(pool_op_kernel_sizes, 1, i_layer, self.num_layers)
                resolution_z = multiply(pool_op_kernel_sizes, 2, i_layer, self.num_layers)
                input_resolution=(
                        pretrain_img_size[0] // resolution_x, 
                        pretrain_img_size[1] // resolution_y,
                        pretrain_img_size[2] // resolution_z)
                # print(i_layer, input_resolution, patch_size)

                layer = BasicLayer(
                    dim = int(embed_dim * 2 ** i_layer),
                    input_resolution=input_resolution,
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size[i_layer],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging,
                    stride=pool_op_kernel_sizes[i_layer + 1]
                    )
                self.add_module(layer_name, layer)

        embed_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers+1)]
        num_features = [int(embed_dim * 2 ** i * 2) for i in range(self.num_layers)]
        self.num_features = num_features
        self.embed_features = embed_features

        # add a norm layer for each output
        for i_modality in range(in_chans):
            for i_layer in out_indices:
                layer = norm_layer(num_features[i_layer])
                layer_name = f'norm{i_modality+1}_{i_layer}'
                self.add_module(layer_name, layer)

    def forward(self, x):
        """Forward function."""
        down = []
        down.append(x)
        x1 = x[:, 0:1, :, :, :]
        x2 = x[:, 1:2, :, :, :]
        x3 = x[:, 2:3, :, :, :]
        
        if self.embed_share_weight:
            x1 = self.patch_embed(x1)
            x2 = self.patch_embed(x2)
            x3 = self.patch_embed(x3)
        else:
            x1 = self.patch_embed1(x1)
            x2 = self.patch_embed2(x2)
            x3 = self.patch_embed3(x3)
       
        Ws, Wh, Ww = x1.size(2), x1.size(3), x1.size(4)
        
        x1 = x1.flatten(2).transpose(1, 2).contiguous()
        x2 = x2.flatten(2).transpose(1, 2).contiguous()
        x3 = x3.flatten(2).transpose(1, 2).contiguous()

        x1 = self.pos_drop1(x1)
        x2 = self.pos_drop2(x2)
        x3 = self.pos_drop3(x3)
        x_o = torch.cat([x1, x2, x3], dim=2)
        x_o = x_o.view(-1, Ws, Wh, Ww, self.embed_features[0]*3).permute(0, 4, 1, 2, 3).contiguous()
        down.append(x_o)

        # stage 1
        x1_od1, Ws1, Wh1, Ww1 = self.layer1_0(x1, x2, x3, Ws, Wh, Ww)
        x2_od1, Ws1, Wh1, Ww1 = self.layer2_0(x2, x1, x3, Ws, Wh, Ww)
        x3_od1, Ws1, Wh1, Ww1 = self.layer3_0(x3, x1, x2, Ws, Wh, Ww)
        x1_o1 = self.norm1_0(x1_od1)
        x2_o1 = self.norm2_0(x2_od1)
        x3_o1 = self.norm3_0(x3_od1)
        x_o1 = torch.cat([x1_o1, x2_o1, x3_o1], dim=2)
        x_o1 = x_o1.view(-1, Ws1, Wh1, Ww1, self.embed_features[1]*3).permute(0, 4, 1, 2, 3).contiguous()
        down.append(x_o1)

        # stage 2
        x1_od2, Ws2, Wh2, Ww2 = self.layer1_1(x1_od1, x2_od1, x3_od1, Ws1, Wh1, Ww1)
        x2_od2, Ws2, Wh2, Ww2 = self.layer2_1(x2_od1, x1_od1, x3_od1, Ws1, Wh1, Ww1)
        x3_od2, Ws2, Wh2, Ww2 = self.layer3_1(x3_od1, x1_od1, x2_od1, Ws1, Wh1, Ww1)
        x1_o2 = self.norm1_1(x1_od2)
        x2_o2 = self.norm2_1(x2_od2)
        x3_o2 = self.norm3_1(x3_od2)
        x_o2 = torch.cat([x1_o2, x2_o2, x3_o2], dim=2)
        x_o2 = x_o2.view(-1, Ws2, Wh2, Ww2, self.embed_features[2]*3).permute(0, 4, 1, 2, 3).contiguous()
        down.append(x_o2)

        # stage 3
        x1_od3, Ws3, Wh3, Ww3 = self.layer1_2(x1_od2, x2_od2, x3_od2, Ws2, Wh2, Ww2)
        x2_od3, Ws3, Wh3, Ww3 = self.layer2_2(x2_od2, x1_od2, x3_od2, Ws2, Wh2, Ww2)
        x3_od3, Ws3, Wh3, Ww3 = self.layer3_2(x3_od2, x1_od2, x2_od2, Ws2, Wh2, Ww2)
        x1_o3 = self.norm1_2(x1_od3)
        x2_o3 = self.norm2_2(x2_od3)
        x3_o3 = self.norm3_2(x3_od3)
        x_o3 = torch.cat([x1_o3, x2_o3, x3_o3], dim=2)
        x_o3 = x_o3.view(-1, Ws3, Wh3, Ww3, self.embed_features[3]*3).permute(0, 4, 1, 2, 3).contiguous()
        down.append(x_o3)

        # stage 4
        x1_od4, S, H, W = self.layer1_3(x1_od3, x2_od3, x3_od3, Ws3, Wh3, Ww3)
        x2_od4, S, H, W = self.layer2_3(x2_od3, x1_od3, x3_od3, Ws3, Wh3, Ww3)
        x3_od4, S, H, W = self.layer3_3(x3_od3, x1_od3, x2_od3, Ws3, Wh3, Ww3)
        x1_o4 = self.norm1_3(x1_od4)
        x2_o4 = self.norm2_3(x2_od4)
        x3_o4 = self.norm3_3(x3_od4)
        x_o4 = torch.cat([x1_o4, x2_o4, x3_o4], dim=2)
        x_o4 = x_o4.view(-1, S, H, W, self.embed_features[4]*3).permute(0, 4, 1, 2, 3).contiguous()
        down.append(x_o4)

        return down


class PreUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str] = 'instance',
        se_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
        """

        super().__init__()

        self.input_channels = in_channels
        self.output_channels = out_channels
        self.transp_conv_init = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            conv_only=True,
            is_transposed=True,
        )

        if se_block:
            self.residual_block = UnetResSEBlock(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.residual_block = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
            )
        
    def forward(self, x):
        x = self.transp_conv_init(x)
        x = self.residual_block(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        upsample_channels: int,
        out_channels: int,
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str] = 'instance',
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.upsample_channels = upsample_channels

        self.transp_conv = get_conv_layer(
            spatial_dims,
            self.input_channels,
            self.input_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.res_block = UnetResBlock(
            spatial_dims,
            self.upsample_channels + self.input_channels,
            self.output_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.res_block(out)
        return out

      
class final_patch_expanding(nn.Module):
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        self.up=nn.ConvTranspose3d(dim, num_class, patch_size, patch_size)
      
    def forward(self,x):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.up(x)
        return x


class PTNet(SegmentationNetwork):
    def __init__(self, crop_size=[8, 320, 320],
                embed_dim=48,
                input_channels=3,
                num_classes=2,
                conv_op=nn.Conv3d,
                norm_name='instance',
                depths=[2, 2, 2, 2],
                num_heads=[4, 8, 16, 32],
                patch_size=[2, 4, 4],
                upsample_size = [1, 2, 2],
                window_size=[4, 4, 8, 4],
                deep_supervision=False,
                pool_op_kernel_sizes=None,
                seg_output_use_bias=False,
                embed_share_weight=False,
                se_block=True):
        super(PTNet, self).__init__()
        
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.upscale_logits_ops = []
        self.upscale_logits_ops.append(lambda x: x)

        self.upsample_size = upsample_size
        self.patch_size = patch_size
        self.embed_ds_size = [patch_size[i] // upsample_size[i] for i in range(len(patch_size))]
        spatial_dims = len(crop_size)
        self.embed_share_weight = embed_share_weight
        self.se_block = se_block

        self.encoder_ds_size = deepcopy(pool_op_kernel_sizes)
        self.encoder_ds_size.insert(0, self.patch_size)
        self.decoder_us_size = deepcopy(pool_op_kernel_sizes)
        self.decoder_us_size.insert(0, self.embed_ds_size)
        self.encoder_embed_size = [embed_dim * 3 * 2 ** i for i in range(len(pool_op_kernel_sizes)+1)]

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.seg_outputs = []

        self.vit = MultimodalSwinT(pretrain_img_size=crop_size,
                               window_size=window_size,
                               embed_dim=embed_dim,
                               patch_size=patch_size,
                               depths=depths,
                               num_heads=num_heads,
                               in_chans=input_channels,
                               pool_op_kernel_sizes=self.encoder_ds_size,
                               embed_share_weight=self.embed_share_weight)

        first_conv_output_channels = self.encoder_embed_size[0] // 3 // 2 # 24
        if self.se_block:
            self.conv_blocks_context.append(UnetResSEBlock(spatial_dims=spatial_dims, in_channels=input_channels,
                                                        out_channels=first_conv_output_channels, kernel_size=3,
                                                        stride=1, norm_name=norm_name))
        else:
            self.conv_blocks_context.append(UnetResBlock(spatial_dims=spatial_dims, in_channels=input_channels,
                                                        out_channels=first_conv_output_channels, kernel_size=3,
                                                        stride=1, norm_name=norm_name))

        for d in range(len(self.encoder_embed_size)):
            in_channels = self.encoder_embed_size[d]
            out_channels = in_channels // 3
            upsample_kernel_size = self.upsample_size
            self.conv_blocks_context.append(PreUpBlock(spatial_dims=spatial_dims, 
                                                       in_channels=in_channels, out_channels=out_channels,  
                                                       upsample_kernel_size=upsample_kernel_size, se_block=self.se_block))
        
        for d in range(len(self.decoder_us_size)):
            in_channels = self.encoder_embed_size[d] // 3
            upsample_channels = self.encoder_embed_size[d] // 3 // 2
            out_channels = self.encoder_embed_size[d] // 3 // 2
            upsample_kernel_size = self.decoder_us_size[d]
            self.conv_blocks_localization.append(UpBlock(spatial_dims=spatial_dims, in_channels=in_channels,
                                                         upsample_channels=upsample_channels, out_channels=out_channels, 
                                                         upsample_kernel_size=upsample_kernel_size))

        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
    
    def forward(self, x):
        skips = []
        seg_outputs = []
        mut_output = self.vit(x)
 
        # encoder
        for d in range(len(self.conv_blocks_context)):
            skips.append(self.conv_blocks_context[d](mut_output[d]))

        # decoder
        for u in range(len(self.conv_blocks_localization)):
            if u == 0:
                enc_x = skips[-1]
                dec_x = skips[-2]
            else:
                dec_x = skips[-(u + 2)]
            enc_x = self.conv_blocks_localization[-(u + 1)](enc_x, dec_x)
            seg_outputs.append(self.seg_outputs[-(u + 1)](enc_x))

        if self.do_ds:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]
        
   

   
