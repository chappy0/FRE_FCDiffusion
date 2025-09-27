from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    # return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    return torch.nn.GroupNorm(num_groups=4, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        # #########print("enter CA")
        super().__init__()
        inner_dim = dim_head * heads
        # ##########print(f"before att:{context_dim, query_dim}")
        context_dim = default(context_dim, query_dim)
        # ##########print(f"which att:{context_dim}")

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.attn_map = None 


    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))


        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        sim = sim.softmax(dim=-1)

        if sim.shape[-1] >= 64:
            sim_for_kl = F.adaptive_avg_pool2d(sim.unsqueeze(1), (64, 64)).squeeze(1)
        else:
            sim_for_kl = sim
        self.attn_map = sim_for_kl
        out = einsum('b i j, b j d -> b i d', sim, v)
        del sim
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)



# def default(value, default_value):
#     return value if value is not None else default_value




class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        # #########print(f"M x shape:{x.shape}")
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)




import torch.nn.functional as F

def l1_norm(x, dim):
    return x / (torch.sum(torch.abs(x), dim=dim, keepdim=True) + 1e-8)

class GroupedDoubleNormalizationAttention(nn.Module):
    def __init__(self, heads, dim_head, num_groups):
        super(GroupedDoubleNormalizationAttention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.num_groups = num_groups

    def forward(self, attn):
        B, H, N, M = attn.shape

        # Step 1: Softmax normalization on attention scores
        attn = F.softmax(attn, dim=2)

        # Step 2: Grouped L1 normalization on attention scores
        # Split the attention scores into multiple groups along the last dimension
        attn_groups = attn.chunk(self.num_groups, dim=3)
        
        # Apply L1 normalization to each group
        normalized_attn_groups = []
        for group in attn_groups:
            group_norm = l1_norm(group, dim=3)
            normalized_attn_groups.append(group_norm)
        
        # Concatenate the normalized groups back together
        attn_normalized = torch.cat(normalized_attn_groups, dim=3)

        return attn_normalized



def trunc_normal_init(param, mean=0, std=1):
    """
    Initialize the input tensor with The Random Truncated Normal (Gaussian) distribution initializer.

    Args:
        param (Tensor): Tensor that needs to be initialized.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
    """
    # Truncated normal distribution is not directly available in PyTorch,
    # but we can use normal distribution and clip the values.
    with torch.no_grad():
        param.uniform_(-2, 2)  # Uniform distribution for initialization
        param.normal_(mean, std)  # Normal distribution
        # Clip values to be within 2 standard deviations
        param.clamp_(-2 * std, 2 * std)

def constant_init(param, value):
    """
    Initialize the `param` with constants.

    Args:
        param (Tensor): Tensor that needs to be initialized.
        value (float): Constant value for initialization.
    """
    with torch.no_grad():
        param.fill_(value)



class GroupedDoubleNormalization(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        B, N, M, d = x.shape
        ########print(f"shape:{x.shape}")
        # Split the last dimension into groups and apply normalization to each group
        x = x.view(B, N, self.num_groups, -1).transpose(1, 2)  # Reshape to (B, num_groups, N, M // num_groups)
        norm_x = F.normalize(x, dim=3)  # Normalize across the group dimension
        ########print(f"norm shape:{norm_x.shape}")
        x = x.transpose(1, 2).view(B, N, -1)  # Reshape to (B, N, M)
        ########print(f"x shape:{x.shape}")
        return norm_x




from functools import partial
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

# import settings
norm_layer = partial(SynchronizedBatchNorm2d, momentum=0.1)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation, 
                               bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out





class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(c_out)
        self.relu = nn.GELU()
        

    def forward(self, x):
        x = self.conv(x)
        ##print(f"ConvBNReLU conv x shape:{x.shape}")
        x = x.permute(1,0,2)
        x = self.bn(x)
        x = self.relu(x)
        ##print(f"ConvBNReLU x shape:{x.shape}")
        x = x.permute(1,0,2)
        return x
    

class RMTBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=7, shift_size=2, mlp_ratio=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.ma_sa = nn.MultiheadAttention(dim, num_heads,batch_first=True)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x):
        B, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        x = x.view(B, H, W, C)

        # attention_mask = torch.zeros((B, 1, N)).to(x.device)
        attention_mask = torch.ones((B*N, B, B), device=x.device)

        # attention_mask = attention_mask.repeat(1, self.num_heads, 1)
        for i in range(H):
            for j in range(W):
                start_i, start_j = max(0, i - self.shift_size), max(0, j - self.shift_size)
                end_i, end_j = min(H, i + self.shift_size + 1), min(W, j + self.shift_size + 1)
                attention_mask[:, :, start_i * W + start_j : end_i * W + end_j] = 1

        # Self-Attention
        attn_output = self.ma_sa(x.view(B, N, C), x.view(B, N, C), x.view(B, N, C), attn_mask=attention_mask)[0]

        # Feed Forward Network
        ffn_output = F.relu(self.fc1(attn_output))
        ffn_output = self.fc2(ffn_output)

        # Add skip connection
        return ffn_output + attn_output






class MultiHeadExternalAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, M=64, coef=1, use_cross_kv=False):
        super(MultiHeadExternalAttention, self).__init__()
        assert in_channels % num_heads == 0, \
            f"in_channels ({in_channels}) should be a multiple of num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.inter_channels = int(M * num_heads * coef)
        self.scale = M ** -0.5
        self.same_in_out_chs = True
        self.use_cross_kv = use_cross_kv

        self.norm = nn.LayerNorm(in_channels)

        self.k = nn.Parameter(torch.randn(self.inter_channels, in_channels, 1, 1) * 0.001)
        self.v = nn.Parameter(torch.randn(self.in_channels, self.inter_channels, 1, 1) * 0.001)
        self.attn_map = None 

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.k, std=0.001)
        nn.init.trunc_normal_(self.v, std=0.001)

    # def _act_sn(self, x):
    #     B, C, N = x.shape
    #     x = x.view(-1, self.inter_channels, N) * self.scale
    #     x = F.softmax(x, dim=1)
    #     return x.view(B, C, N)
    def _act_sn(self, x):
        B, C, N = x.shape
        x = x.view(-1, self.inter_channels, N) * self.scale
        attn = F.softmax(x, dim=1)              
        self.attn_map = attn.view(B, C, N)      
        return attn.view(B, C, N)

    def _act_dn(self, x):
        B, C, N = x.shape
        x = x.view(B, self.num_heads, self.inter_channels // self.num_heads, N)
        attn = F.softmax(x * self.scale, dim=-1)  
        self.attn_map = attn                       
        x = attn / (torch.sum(attn, dim=2, keepdim=True) + 1e-6)
        return x.view(B, self.inter_channels, N)


    def forward(self, x, cross_k=None, cross_v=None):
        """
        Args:
            x (Tensor): The input tensor with shape (B, N, C).
            cross_k (Tensor, optional): Cross-key tensor.
            cross_v (Tensor, optional): Cross-value tensor.
        """
        B, N, C = x.shape

        # Normalize input
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous()  # (B, C, N)

        if not self.use_cross_kv:
            x = F.conv2d(x.view(B, C, 1, N), self.k, stride=2 if not self.same_in_out_chs else 1)  # (B, inter_C, 1, N)
            x = x.squeeze(2)  # (B, inter_C, N)
            x = self._act_dn(x)
            x = F.conv2d(x.unsqueeze(2), self.v).squeeze(2)  # (B, out_C, N)
        else:
            assert cross_k is not None and cross_v is not None, "cross_k and cross_v must not be None when use_cross_kv"
            x = x.view(1, -1, 1, N)  # (1, B*C, 1, N)
            x = F.conv2d(x, cross_k, groups=B)  # (1, B*144, 1, N)
            x = self._act_sn(x.squeeze(2))
            x = F.conv2d(x.unsqueeze(2), cross_v, groups=B).squeeze(2)  # (1, B*C, 1, N)
            x = x.view(B, self.in_channels, N)  # (B, C, N)

        return x.permute(0, 2, 1)  # (B, N, C)


def bn2d(in_channels, bn_mom=0.1, lr_mult=1.0, **kwargs):
    assert 'bias' not in kwargs, "bias must not be in kwargs"
    return nn.BatchNorm2d(in_channels, momentum=bn_mom, **kwargs)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='none', ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction, 
                                   ignore_index=ignore_index)

    def forward(self, inputs, targets):
        loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return loss.mean(dim=2).mean(dim=1)


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(c_out)
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        #print(f"before bn:{x.shape}")
        x = x.permute(2,0,1)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
        



class ff_EA2(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(ff_EA2, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_channels)

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

    def forward(self, x):

        x = x.reshape(x.size(0), -1)
        #print(f"reshape x shape:{x.shape}")
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        out_channels = self.fc2.out_features
        batch_size, height, width = x.size(0), int(out_channels ** 0.5), out_channels // height
        x = x.view(batch_size, out_channels, height, width)

        x = self.depthwise_conv(x)
        
        return x

class ff_EA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc1 = nn.Sequential(
            ConvBNReLU(dim, dim, 3, 1, 1, 1),
            nn.Dropout2d(p=0.1))
        self.fc2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.permute(1,0,2)
        x = self.fc2(x)
        #print(f"after fc2 shape:{x.shape}")  #c,b,n
        x = x.permute(1,2,0)
        return x


class ff_RMT(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x):
        ffn_output = F.relu(self.fc1(x))
        ffn_output = self.fc2(ffn_output)
        return ffn_output
    

class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head,time_embed_dim, use_dynamic_hybrid_attention=False, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False,use_external_attention=False,use_RMT=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn

        # self.ff_SA = nn.Sequential(ConvBNReLU(dim, dim, 3, 1, 1, 1),nn.Dropout2d(p=0.1),nn.Conv2d(dim, dim, 1))
        self.ff_EA = ff_EA(dim)
        # self.ff_EA = ff_EA2(dim,128,dim)
        # self.fc2 = nn.Conv2d(dim, dim, 1)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        if not self.disable_self_attn:
            if use_dynamic_hybrid_attention:
                print("Using Dynamic Hybrid Attention")
                self.attn1 = DynamicHybridAttention(query_dim=dim, n_heads=n_heads, d_head=d_head, time_embed_dim=time_embed_dim,dropout=dropout)

            elif use_external_attention: 
                # print(f"dim:{dim},heads:{n_head},d_head:{d_head}")
                print("use_external_attention")
                # self.attn1 = MultiHeadExternalAttention(dim,8,64)
                self.attn1 = MultiHeadExternalAttention(dim,n_heads,d_head)  # need to update
            elif use_RMT:
                self.attn1 = RMTBlock(dim,n_heads,d_head)
            else:
                self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
                # self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        else:
            self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
            
        
        if context_dim is None:
            if use_external_attention: 
                self.attn2 = MultiHeadExternalAttention(dim)
                # self.attn2 = ExternalAttention(dim)
                # self.fc1 = nn.Sequential(ConvBNReLU(dim, dim, 3, 1, 1, 1),nn.Dropout2d(p=0.1))
                # self.fc2 = nn.Conv2d(dim, dim, 1)
                # self.attn2 = ExternalAttention(C_in=dim,heads=1, M_k=d_head, M_v= d_head)
                # self.attn2 = ExternalAttention(dim, context_dim, heads=8, dim_head=64, dropout=0.1)
            else:
                self.attn2 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        
        else:
            self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None,t_embedding=None):
        return checkpoint(self._forward, (x, context, t_embedding), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, t_embedding=None):
        # print("forward")
        if isinstance(self.attn1, DynamicHybridAttention):
        # Pass the t_embedding to the DHA module
            # print('forward using DynamicHybridAttention')
        # Get the output from the attention module
            attn1_output = self.attn1(self.norm1(x), t_embedding=t_embedding)

            # Check if the output is a tuple (training mode) or a single tensor (inference mode)
            if isinstance(attn1_output, tuple):
                # Unpack the tuple and use the first element for the residual connection
                output, sa_map, ea_map = attn1_output
                x = output + x
            else:
                # Handle the case where only the output tensor is returned
                x = attn1_output + x
        ##print(f"BT xshape:{x.shape}")
        elif isinstance(self.attn1, MultiHeadExternalAttention):
            # print(f"forward with EA")
            x = self.attn1(self.norm1(x)) + x
            # x = x.permute(2,1,0)
            # #print(f"after attn1 x shape:{x.shape}")
            # # x = self.ff_SA(x).permute(2,1,0)+ x.permute(2,1,0)
            # x = self.ff(x)+ x.permute(2,1,0)
            
            # x += F.interpolate(x,  mode='nearest',scale_factor=1)  #, align_corners=True
            #########print(f"after attn1 x shape:{x.shape}")
            # x=x.permute(2,1,0)
            # x = self.fc1(x)
            # x = self.fc2(x)
        elif isinstance(self.attn1, RMTBlock):
            x = self.attn1(self.norm1(x)) + x

        else:
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x


        if isinstance(self.attn2, MultiHeadExternalAttention):
            attn2_output = self.attn2(self.norm2(x))
            ##########print(f"after attn2 x shape:{attn2_output.shape}")

            x = attn2_output + x
            #########print(f"after attn2+x x shape:{attn2_output.shape}")
            x = self.ff_EA(self.norm3(x)) + x
            # x += F.interpolate(x,  mode='nearest', align_corners=True)  #bilinear

            
        elif self.attn2 is not None and context is not None:
            attn2_output = self.attn2(self.norm2(x), context=context)
            x = attn2_output + x  


            x = self.ff(self.norm3(x)) + x  
        else:
            attn2_output = self.attn2(self.norm2(x))
            x = attn2_output + x  

            x = self.ff(self.norm3(x)) + x  
        return x


class DynamicHybridAttention(nn.Module):

    def __init__(self, query_dim, n_heads, d_head, time_embed_dim, 
                 min_sa_tokens=16, dropout=0.):
        super().__init__()
        self.query_dim = query_dim
        self.min_sa_tokens = min_sa_tokens  
        

        self.sa_attn = CrossAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ea_attn = MultiHeadExternalAttention(in_channels=query_dim, num_heads=n_heads, M=d_head)

        self.gating_network = nn.Sequential(
            nn.Linear(time_embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, query_dim) 
        )
        
        self.global_alpha = nn.Parameter(torch.tensor(0.5)) 

    def forward(self, x, t_embedding=None):
        batch_size, seq_len, _ = x.shape
    
        if t_embedding is None:
            return self.sa_attn(x)

        local_scores = self.gating_network(t_embedding)  # (batch_size, query_dim)
        token_importance = torch.einsum("bqd,bd->bq", x, local_scores)  # (batch_size, seq_len)

        k = max(self.min_sa_tokens, int(seq_len * 0.1))  
        _, top_indices = torch.topk(token_importance, k, dim=1)  # (batch_size, k)

        expanded_indices = top_indices.unsqueeze(-1).repeat(1, 1, self.query_dim)
        x_important = torch.gather(x, dim=1, index=expanded_indices)  # (batch_size, k, query_dim)
        
        sa_important_out = self.sa_attn(x_important)  # (batch_size, k, query_dim)
    
        sa_out = torch.zeros_like(x)  
        sa_out = sa_out.scatter(dim=1, index=expanded_indices, src=sa_important_out)
        ea_out = self.ea_attn(x)  # (batch_size, seq_len, query_dim)
        output = ea_out + self.global_alpha * sa_out
        
        self.sa_indices_cache = top_indices.detach()
        self.global_alpha_cache = self.global_alpha.detach()
        

        sa_attn = getattr(self.sa_attn, 'attn_map', None)
        ea_attn = getattr(self.ea_attn, 'attn_map', None)
        if sa_attn is not None:
            self.sa_attn.attn_map = None
        if ea_attn is not None:
            self.ea_attn.attn_map = None
        
        if self.training and sa_attn is not None and ea_attn is not None:
            return output, sa_attn, ea_attn
        return output



class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,time_embed_dim,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True,  use_external_attention=False,
                 use_dynamic_hybrid_attention=False):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d] if context_dim else None,
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint,use_external_attention=use_external_attention,
                                   time_embed_dim=time_embed_dim,use_dynamic_hybrid_attention=use_dynamic_hybrid_attention)
             for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear



    def forward(self, x, emb, context=None):
        if not isinstance(context, list):
            context = [context]  # b, n, context_dim
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)  # b, hw, inner_dim
            
        # Pass the 'emb' down to the transformer blocks as 't_embedding'
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i] if context is not None and i < len(context) else None, t_embedding=emb)

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
