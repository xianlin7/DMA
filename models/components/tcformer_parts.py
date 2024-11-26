import math
import warnings
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import torch
from mmcv.utils import get_logger
from mmcv.runner import _load_checkpoint, load_state_dict
import re
import logging

#---------------------------------------------tcformer_utils---------------------------------------------
def get_grid_index(init_size, map_size, device):
    """For each initial grid, get its index in the feature map.
    Returns:
        idx (LongTensor[B, N_init]): index in flattened feature map.

    Args:
        init_grid_size(list[int] or tuple[int]): initial grid resolution in
            format [H_init, W_init].
        map_size(list[int] or tuple[int]): feature map resolution in format
            [H, W].
        device: the device of output
    """
    H_init, W_init = init_size
    H, W = map_size
    idx = torch.arange(H * W, device=device).reshape(1, 1, H, W)
    idx = F.interpolate(idx.float(), [H_init, W_init], mode='nearest').long()
    return idx.flatten()


def token2map(token_dict):
    """Transform vision tokens to feature map. This function only
    works when the resolution of the feature map is not higher than
    the initial grid structure.
    Returns:
        x_out (Tensor[B, C, H, W]): feature map.

    Args:
        token_dict (dict): dict for token information.
    """

    x = token_dict['x']
    H, W = token_dict['map_size']
    H_init, W_init = token_dict['init_grid_size']
    idx_token = token_dict['idx_token']
    B, N, C = x.shape
    N_init = H_init * W_init
    device = x.device

    if N_init == N and N == H * W:
        # for the initial tokens with grid structure, just reshape
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    # for each initial grid, get the corresponding index in
    # the flattened feature map.
    idx_hw = get_grid_index(
        [H_init, W_init], [H, W], device=device)[None, :].expand(B, -1) #[B N_init]∈(0, hw)
    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
    value = x.new_ones(B * N_init)

    # choose the way with fewer flops.
    if N_init < N * H * W:
        # use sparse matrix multiplication
        # Flops: B * N_init * (C+2)
        idx_hw = idx_hw + idx_batch * H * W
        idx_tokens = idx_token + idx_batch * N
        coor = torch.stack([idx_hw, idx_tokens], dim=0).reshape(2, B * N_init)

        # torch.sparse do not support fp16
        with torch.cuda.amp.autocast(enabled=False):
            # torch.sparse do not support gradient for
            # sparse tensor, so we detach it
            value = value.detach().float()

            # build a sparse matrix with the shape [B * H * W, B * N]
            A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))

            # normalize the weight for each row
            all_weight = A @ x.new_ones(B * N, 1).type(torch.float32) + 1e-6
            value = value / all_weight[idx_hw.reshape(-1), 0]

            # update the matrix with normalize weight
            A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))

            # sparse matrix multiplication
            x_out = A @ x.reshape(B * N, C).type(torch.float32)  # [B*H*W, C]

    else:
        # use dense matrix multiplication
        # Flops: B * N * H * W * (C+2)
        coor = torch.stack([idx_batch, idx_hw, idx_token], dim=0).reshape(3, B * N_init)

        # build a matrix with shape [B, H*W, N]
        A = torch.sparse.FloatTensor(coor, value, torch.Size([B, H * W, N])).to_dense()
        # normalize the weight
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        x_out = A @ x  # [B, H*W, C]

    x_out = x_out.type(x.dtype)
    x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    return x_out

def map2token(feature_map, token_dict):
    """Transform feature map to vision tokens. This function only
    works when the resolution of the feature map is not higher than
    the initial grid structure.

    Returns:
        out (Tensor[B, N, C]): token features.

    Args:
        feature_map (Tensor[B, C, H, W]): feature map.
        token_dict (dict): dict for token information.
    """
    idx_token = token_dict['idx_token']
    N = token_dict['token_num']
    H_init, W_init = token_dict['init_grid_size']
    N_init = H_init * W_init

    # agg_weight = token_dict['agg_weight'] if 'agg_weight' in token_dict.keys() else None
    agg_weight = None  # we do not use the weight value here

    B, C, H, W = feature_map.shape
    device = feature_map.device

    if N_init == N and N == H * W:
        # for the initial tokens with grid structure, just reshape
        return feature_map.flatten(2).permute(0, 2, 1).contiguous()

    idx_hw = get_grid_index(
        [H_init, W_init], [H, W], device=device)[None, :].expand(B, -1) # [b N_init] the position of init map correspond to the index of now feature map e.g.[0 0 1 1 ... 1023 1023]

    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init) # [b N_init] e.g.[[0 0 ... 0],[1 1 ... 1]]
    if agg_weight is None:
        value = feature_map.new_ones(B * N_init) #[b*N_init] e.g.[8192] [1 1 ... 1]
    else:
        value = agg_weight.reshape(B * N_init).type(feature_map.dtype)

    # choose the way with fewer flops.
    if N_init < N * H * W:
        # use sparse matrix multiplication
        # Flops: B * N_init * (C+2)
        idx_token = idx_token + idx_batch * N #[b N], each token in the same batch have different index e.g.[[0 1 2 ... 4095],[4096 4097 ... 8191]]
        idx_hw = idx_hw + idx_batch * H * W # [b H * W], each position in the same batch have different index correspond to the feature map e.g.[[0 0 1 1 ... 1023 1023],[1024 1024 ... 2047 2047]]
        indices = torch.stack([idx_token, idx_hw], dim=0).reshape(2, -1) # [2 b*N]

        # torch.sparse do not support fp16
        with torch.cuda.amp.autocast(enabled=False):
            # sparse mm do not support gradient for sparse matrix
            value = value.detach().float()
            # build a sparse matrix with shape [B*N, B*H*W]
            A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W)) #[b*N_init], in tensor A [B*N, B*H*W], the pixel value in indices positions are the value
            # normalize the matrix
            all_weight = A @ torch.ones(
                [B * H * W, 1], device=device, dtype=torch.float32) + 1e-6 # [B*N 1]
            value = value / all_weight[idx_token.reshape(-1), 0]

            A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W)) # A [B*N, B*H*W]
            # out: [B*N, C]
            out = A @ feature_map.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C).float() # the feature of each token is the result of weighting feature map
    else:
        # use dense matrix multiplication
        # Flops: B * N * H * W * (C+2)
        indices = torch.stack([idx_batch, idx_token, idx_hw], dim=0).reshape(3, -1)
        value = value.detach()  # To reduce the training time, we detach here.
        A = torch.sparse_coo_tensor(indices, value, (B, N, H * W)).to_dense()
        # normalize the matrix
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        out = A @ feature_map.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

    out = out.type(feature_map.dtype)
    out = out.reshape(B, N, C)
    return out


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C], e.g. [B N N], the distance between two points
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape) # b, cluster_num
    view_shape[1:] = [1] * (len(view_shape) - 1) # b, 1
    repeat_shape = list(idx.shape) # b, cluster_num
    repeat_shape[0] = 1 # 1, cluster_num
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # [b cluster_num], e.g., [[0 0 ... 0 0],[1 1 ... 1]]
    new_points = points[batch_indices, idx, :] # [b cluster_num N]
    return new_points

def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """

    x = token_dict['x'] # [B N C]
    idx_token = token_dict['idx_token'] # [B N], e.g., [[0 1 2 ... N-1],[0 1 2 ... N-1]]
    agg_weight = token_dict['agg_weight'] # [B N 1]

    B, N, C = x.shape
    if token_weight is None: # [B N 1] the fc predicted important score
        token_weight = x.new_ones(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None] # [B 1]
    idx = idx_cluster + idx_batch * cluster_num #[b N], each cluster point in the same batch have different id

    all_weight = token_weight.new_zeros(B * cluster_num, 1) # [B*cluster_num 1]
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6 # [B*cluster_num 1]
    norm_weight = token_weight / all_weight[idx] # [B N 1]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C) 
    source = x * norm_weight # [B N C]
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1) #[B N]
    weight_t = index_points(norm_weight, idx_token) # [B N 1]
    agg_weight_new = agg_weight * weight_t # [B N 1]
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    out_dict['map_size'] = token_dict['map_size']
    out_dict['init_grid_size'] = token_dict['init_grid_size']
    out_dict['idx_token'] = idx_token_new
    out_dict['agg_weight'] = agg_weight_new
    return out_dict
