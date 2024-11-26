from operator import index
from traceback import print_tb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.visualization import vis_cluster_results, vis_grid_results 

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2in(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    

class qkvAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        input_size = (32, 32)
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q= nn.Linear(dim, dim, bias=qkv_bias)
        self.k= nn.Linear(dim, dim, bias=qkv_bias)
        self.v= nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, kv):
        B, Nq, _ = q.shape
        Nk = kv.shape[1]
        q = self.q(q).reshape(B, Nq, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B*self.num_heads, Nq, -1)
        k = self.k(kv).reshape(B, Nk, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B*self.num_heads, Nk, -1)
        v = self.v(kv).reshape(B, Nk, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B*self.num_heads, Nk, -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        #attn = attn.softmax(dim=-1)
        attn = softmax_one(attn, dim=-1)
        x = (attn @ v).view(B, self.num_heads, Nq, -1).permute(0, 2, 1, 3).reshape(B, Nq, -1)
        x = self.proj(x)

        return x


class vitAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, qx, kx):
        q = self.to_q(qx)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(kx).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def cluster_dependency(prototypes, x):
    with torch.no_grad():
        dim= prototypes.shape[2]
        # --- normalize the prototypes --------------
        dist_matrix = torch.cdist(prototypes, x) / (dim ** 0.5) # [B P N]
        index_cluster = dist_matrix.argmin(dim=1) # [B N]
    return index_cluster, dist_matrix


class DependencyMerge(nn.Module):
    def __init__(self, dim=512, n_classes=2, k_class0=1):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.sig = nn.Sigmoid()
        self.dim = dim
        self.n_classes = n_classes
        self.k_class0 = k_class0

    def forward(self, prototypes, x0):
        x = torch.cat([prototypes, x0], dim=1)
        B, N, C = x.shape
        cluster_num = prototypes.shape[1]
        n_cprototypes = cluster_num//self.n_classes
        s_weight = self.sig(self.score(x))

        idx_cluster, dist_matrix = cluster_dependency(prototypes, x) # if choose3, the following 2 lines shoud add a #
        dist_matrix_group = rearrange(dist_matrix, 'B (C P) N -> B N C P', C=self.n_classes)
        dist_matrix_group = (-dist_matrix_group).exp()

        c_weight = torch.sum(dist_matrix_group, dim=-1)/n_cprototypes # [B N c]
        c_weight_all = torch.sum(c_weight, dim=-1) + 1e-6 # [B N]
        #print("cwa max:", torch.max(c_weight_all), "min:", torch.min(c_weight_all))

        c_weight = c_weight / c_weight_all[:, :, None] # [B N c]
        idx_batch = torch.arange(B, device=x.device)[:, None] # [B 1]
        idx = idx_cluster + idx_batch * (cluster_num) # [B N]

        idx_c0 = torch.arange(B, device=x.device)[:, None].repeat(1, N)
        idx_c1 = torch.arange(N, device=x.device)[None, :].repeat(B, 1) # [1 N]
        idx_cluster_c = idx_cluster//n_cprototypes
        c_weight = c_weight[idx_c0, idx_c1, idx_cluster_c][:, :, None] # [B N 1]

        all_c_weight = c_weight.new_zeros(B*cluster_num, 1)
        all_c_weight.index_add_(dim=0, index=idx.reshape(B*N), source=c_weight.reshape(B*N, 1))
        all_c_weight = all_c_weight + 1e-6
        #print("cw max:", torch.max(all_c_weight), "min:", torch.min(all_c_weight))
        norm_c_weight = c_weight / all_c_weight[idx] # [B N 1]
        #print("ncw max:", torch.max(norm_c_weight), "min:", torch.min(norm_c_weight))

        all_s_weight = s_weight.new_zeros(B*cluster_num, 1)
        all_s_weight.index_add_(dim=0, index=idx.reshape(B*N), source=s_weight.reshape(B*N, 1))
        all_s_weight = all_s_weight + 1e-6
        norm_s_weight = s_weight / all_s_weight[idx] # [B N 1]
        #print("nsw max:", torch.max(norm_s_weight), "min:", torch.min(norm_s_weight))

        prototypes_merged = prototypes.new_zeros(B*cluster_num, C) 
        source = x * (0.5*norm_c_weight + 0.5*norm_s_weight) # [B N C]
        #print("c max:", torch.max(x), "min:", torch.min(x))
        prototypes_merged.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(prototypes.dtype))
        prototypes_merged = prototypes_merged.reshape(B, cluster_num, C)
        #print("pm max:", torch.max(prototypes_merged), "min:", torch.min(prototypes_merged))
      
        return prototypes_merged, idx_cluster



class DSTransformerIP(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., n_classes=2, n_cprototypes=64, f_merge=3, input_size=(32, 32), pdepth=2, k_class0=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.prototype_layers = nn.ModuleList([])
        self.f_merge=f_merge
        self.n_cprototypes=n_cprototypes
        self.n_classes = n_classes
        self.pool1d = nn.MaxPool1d(2, stride=2)
        self.dim = dim
        self.depth = depth
        self.input_size = input_size
        self.k_class0 = k_class0
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2in(dim, vitAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                DependencyMerge(dim=dim, n_classes=n_classes, k_class0=k_class0),
            ]))
        # ------------- protypes fission network -------------
        self.prototypes = nn.Parameter(torch.randn(1, n_classes*(n_cprototypes//16), dim))
        self.tokenfission1 = nn.Linear(dim, 2*dim)
        self.tokenfission2 = nn.Linear(dim, 2*dim)
        self.token2image2 = nn.ModuleList([
                PreNorm2in(dim, qkvAttention(dim, num_heads=heads, input_size=input_size)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ])
        self.tokenfission3 = nn.Linear(dim, 2*dim)
        self.tokenfission4 = nn.Linear(dim, 2*dim)
        self.token2image4 = nn.ModuleList([
                PreNorm2in(dim, qkvAttention(dim, num_heads=heads, input_size=input_size)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ])
     
    def forward(self, x):

        B = x.shape[0]
        prototypes = self.prototypes.repeat(B, 1, 1) # [B ck1 d]

        #init the prototypes features by feature maps
        prototypes = self.tokenfission1(prototypes).reshape(B, self.n_classes*(self.n_cprototypes//16), 2, self.dim).reshape(B, self.n_classes*(self.n_cprototypes//8), self.dim)
        prototypes = self.tokenfission2(prototypes).reshape(B, self.n_classes*(self.n_cprototypes//8), 2, self.dim).reshape(B, self.n_classes*(self.n_cprototypes//4), self.dim)
        pattn, pff = self.token2image2
        prototypes = pattn(prototypes, x) + prototypes
        prototypes = pff(prototypes) + prototypes
        prototypes = self.tokenfission3(prototypes).reshape(B, self.n_classes*(self.n_cprototypes//4), 2, self.dim).reshape(B, self.n_classes*(self.n_cprototypes//2), self.dim)
        prototypes = self.tokenfission4(prototypes).reshape(B, self.n_classes*(self.n_cprototypes//2), 2, self.dim).reshape(B, self.n_classes*self.n_cprototypes, self.dim)
        pattn, pff = self.token2image4
        prototypes = pattn(prototypes, x) + prototypes
        prototypes = pff(prototypes) + prototypes

        for id_layer, (attn, ff, dm) in enumerate(self.layers):
            #----- cluster dependecy ------------
            prototypes, id_cluster = dm(prototypes, x) 

            # vis_grid_results(x, prototypes, id_cluster, id_layer, self.n_classes, self.n_cprototypes, prototypes.shape[1]//2, prototypes.shape[1]//2)
            # vis_cluster_results(x, prototypes, id_cluster, id_layer, self.n_classes, self.n_cprototypes, prototypes.shape[1]//2, prototypes.shape[1]//2)

            x = attn(x, prototypes) + x
            x = ff(x) + x

            if (id_layer+1) % self.f_merge == 0 and (id_layer+1)<self.depth:
                prototypes = self.pool1d(prototypes.permute(0, 2, 1)).permute(0, 2, 1)

        # ----------------- generate the deep supurvise feature map -----------------
        B, K, D = prototypes.shape
        deep_pred = []
        if self.training:
            deep_pred = torch.matmul(prototypes, x.transpose(-1, -2)) * (D ** -0.5) # [B K N]
            deep_pred = deep_pred.reshape(B, K//self.n_classes, self.n_classes, -1)
            deep_pred = torch.mean(deep_pred, dim=1) # [B C N]
            deep_pred = deep_pred.reshape(B, self.n_classes, self.input_size[0], self.input_size[1])
            return x, deep_pred, self.prototypes.reshape(1, self.n_classes, self.n_cprototypes//16, self.dim), prototypes.reshape(B, self.n_classes, K//self.n_classes, D)
            #return x, deep_pred, self.prototypes.reshape(1, self.n_classes, self.n_cprototypes, self.dim), prototypes.reshape(B, self.n_classes, K//self.n_classes, D)
        else:
            #return x, deep_pred, self.prototypes, prototypes
            return x, deep_pred, self.prototypes.reshape(1, self.n_classes, self.n_cprototypes//16, self.dim), prototypes.reshape(B, self.n_classes, K//self.n_classes, D)

