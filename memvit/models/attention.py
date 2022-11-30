#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


from functools import partial

import memvit.utils.logging as logging
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from memvit.models.common import DropPath, Mlp
from torch.nn.init import trunc_normal_

logger = logging.get_logger(__name__)


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None, shift_t=0):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    tensor = pool(tensor)
    if shift_t > 0:
        tensor = tensor[:, :, :-shift_t]

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


def get_rel_pos(rel_pos, d):
    """
    Get relative positional embeddings based on the size d.
    """
    if isinstance(d, int):
        ori_d = rel_pos.shape[0]
        if ori_d == d:
            return rel_pos
        else:
            # Interpolate rel pos.
            new_pos_embed = F.interpolate(
                rel_pos.reshape(1, ori_d, -1).permute(0, 2, 1),
                size=d,
                mode="linear",
            )

            return new_pos_embed.reshape(-1, d).permute(1, 0)


def get_spatial_embeddings(q_shape, k_shape, rel_pos_h, rel_pos_w):
    """
    Sample the relative positional embedding matrix.
    """
    _, q_h, q_w = q_shape
    _, k_h, k_w = k_shape

    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)
    # Intepolate rel pos if needed.
    rel_pos_h = get_rel_pos(rel_pos_h, dh)

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio

    if dh == dw:
        rel_pos_w = get_rel_pos(rel_pos_w, dw)
        q_w_ratio = max(k_w / q_w, 1.0)
        k_w_ratio = max(q_w / k_w, 1.0)
        dist_w = (
            torch.arange(q_w)[:, None] * q_w_ratio
            - torch.arange(k_w)[None, :] * k_w_ratio
        )
        dist_w += (k_w - 1) * k_w_ratio
    elif dw > dh:
        rel_pos_w = get_rel_pos(rel_pos_w, dh)
        dist_w = (
            torch.arange(q_w)[:, None] * q_h_ratio
            - torch.arange(k_w)[None, :] * k_h_ratio
        )
        dist_w += dist_h[0, 0]
        dist_w = torch.clamp(dist_w, dist_h.min(), dist_h.max())

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]
    return Rh, Rw


def get_spatial_rel_score(r_q_h, r_q_w, Rh, Rw, B, n_head, q_shape, k_shape):
    """
    Compute the final score with spatial positional embedding.
    """
    q_t, q_h, q_w = q_shape
    _, k_h, k_w = k_shape

    # [q_h, B*H*q_t*q_w, dim] * [q_h, dim, k_h] = [q_h, B*H*q_t*q_w, k_h] -> [B*H*q_t*q_w, q_h, k_h]
    rel_h = torch.matmul(r_q_h, Rh.permute(0, 2, 1)).transpose(0, 1)
    # [q_w, B*H*q_t*q_h, dim] * [q_w, dim, k_w] = [q_w, B*H*q_t*q_h, k_w] -> [B*H*q_t*q_h, q_w, k_w]
    rel_w = torch.matmul(r_q_w, Rw.permute(0, 2, 1)).transpose(0, 1)

    # [B*H*q_t*q_w, q_h, k_h] -> [B, H, q_t, qh, qw, k_h]
    rel_h = rel_h.view(B, n_head, q_t, q_w, q_h, k_h).permute(0, 1, 2, 4, 3, 5)
    # [B*H*q_t*q_h, q_w, k_w] -> [B, H, q_t, qh, qw, k_w]
    rel_w = rel_w.view(B, n_head, q_t, q_h, q_w, k_w)
    return rel_h, rel_w


def cal_rel_pos_spatial(
    attn,
    q,
    has_cls_embed,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
    mem_k_shape,
    rel_pos_mem_h,
    rel_pos_mem_w,
    mem_k_len,
):
    """
    Main Spatial Relative Positional Embeddings function.
    """

    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape

    if mem_k_shape is not None:
        Rh_mem, Rw_mem = get_spatial_embeddings(
            q_shape, mem_k_shape, rel_pos_mem_h, rel_pos_mem_w
        )

    Rh, Rw = get_spatial_embeddings(q_shape, k_shape, rel_pos_h, rel_pos_w)
    B, n_head, _, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    # [B, H, q_t, q_h, q_w, dim] -> [q_h, B, H, q_t, q_w, dim] -> [q_h, B*H*q_t*q_w, dim]
    r_q_h = r_q.permute(3, 0, 1, 2, 4, 5).reshape(q_h, B * n_head * q_t * q_w, dim)
    # [B, H, q_t, q_h, q_w, dim] -> [q_w, B, H, q_t, q_h, dim] -> [q_w, B*H*q_t*q_h, dim]
    r_q_w = r_q.permute(4, 0, 1, 2, 3, 5).reshape(q_w, B * n_head * q_t * q_h, dim)

    if mem_k_shape is not None:
        rel_h_mem, rel_w_mem = get_spatial_rel_score(
            r_q_h, r_q_w, Rh_mem, Rw_mem, B, n_head, q_shape, mem_k_shape
        )
    rel_h, rel_w = get_spatial_rel_score(r_q_h, r_q_w, Rh, Rw, B, n_head, q_shape, k_shape)

    mem_kv_end = 0
    if mem_k_shape is not None:
        mem_k_t, mem_k_h, mem_k_w = mem_k_shape

        for kv_idx in range(mem_k_len):
            kv_size = numpy.prod(mem_k_shape) // mem_k_len + sp_idx
            mem_kv_start = sp_idx + kv_idx * kv_size
            mem_kv_end = (kv_idx + 1) * kv_size

            attn[:, :, sp_idx:, mem_kv_start:mem_kv_end] = (
                attn[:, :, sp_idx:, mem_kv_start:mem_kv_end].view(
                    B, -1, q_t, q_h, q_w, mem_k_t // mem_k_len, mem_k_h, mem_k_w
                )
                + rel_h_mem[:, :, :, :, :, None, :, None]
                + rel_w_mem[:, :, :, :, :, None, None, :]
            ).view(
                B,
                -1,
                q_t * q_h * q_w,
                (mem_k_t // mem_k_len) * mem_k_h * mem_k_w,
            )

    kv_start = mem_kv_end + sp_idx
    kv_end = mem_kv_end + sp_idx + numpy.prod(k_shape)

    assert kv_end == attn.shape[-1]
    attn[:, :, sp_idx:, kv_start:kv_end] = (
        attn[:, :, sp_idx:, kv_start:kv_end].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel_h[:, :, :, :, :, None, :, None]
        + rel_w[:, :, :, :, :, None, None, :]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


def cal_rel_pos_temporal(
    attn,
    q,
    has_cls_embed,
    q_shape,
    k_shape,
    rel_pos_t,
    mem_k_shape,
    rel_pos_mem_t,
    mem_k_stride,
    mem_k_len,
):
    """
    Main Temporal Relative Positional Embeddings function.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape

    if mem_k_shape is not None:
        mem_k_t, mem_k_h, mem_k_w = mem_k_shape
        dist_t_mem = (
            torch.arange(q_t)[:, None] - torch.arange(mem_k_t)[None, :] * mem_k_stride
        )
        dist_t_mem = dist_t_mem - dist_t_mem[0, -1]

        Rt_mem = rel_pos_mem_t[dist_t_mem.long()]

    dist_t = torch.arange(q_t)[:, None] - torch.arange(k_t)[None, :]
    dist_t = dist_t - dist_t[0, -1]
    Rt = rel_pos_t[dist_t.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    # [B, H, q_t, q_h, q_w, dim] -> [q_t, B, H, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
    r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(q_t, B * n_head * q_h * q_w, dim)

    if mem_k_shape is not None:
        # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
        rel_mem = torch.matmul(r_q, Rt_mem.transpose(1, 2)).transpose(0, 1)
        # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
        rel_mem = rel_mem.view(B, n_head, q_h, q_w, q_t, mem_k_t).permute(
            0, 1, 4, 2, 3, 5
        )

    # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
    rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
    rel = rel.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    mem_kv_end = 0
    if mem_k_shape is not None:
        rel_mem = torch.chunk(rel_mem, mem_k_len, dim=5)
        for kv_idx in range(mem_k_len):
            kv_size = numpy.prod(mem_k_shape) // mem_k_len + sp_idx
            mem_kv_start = sp_idx + kv_idx * kv_size
            mem_kv_end = (kv_idx + 1) * kv_size

            attn[:, :, sp_idx:, mem_kv_start:mem_kv_end] = (
                attn[:, :, sp_idx:, mem_kv_start:mem_kv_end].view(
                    B, -1, q_t, q_h, q_w, mem_k_t // mem_k_len, mem_k_h, mem_k_w
                )
                + rel_mem[kv_idx][:, :, :, :, :, :, None, None]
            ).view(
                B,
                -1,
                q_t * q_h * q_w,
                (mem_k_t // mem_k_len) * mem_k_h * mem_k_w,
            )

    kv_start = mem_kv_end + sp_idx
    kv_end = mem_kv_end + sp_idx + numpy.prod(k_shape)

    assert kv_end == attn.shape[-1]
    attn[:, :, sp_idx:, kv_start:kv_end] = (
        attn[:, :, sp_idx:, kv_start:kv_end].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel[:, :, :, :, :, :, None, None]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


def get_conv_q(tensor, thw_shape, func, has_cls_embed=True):
    if has_cls_embed:
        _, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    tensor = func(tensor)
    tensor = tensor.reshape(B, N, C, T * H * W).transpose(2, 3)

    if has_cls_embed:
        tensor = torch.cat(
            (torch.zeros((B, N, 1, C), device=tensor.device), tensor), dim=2
        )

    return tensor


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        input_size,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        # If True, perform pool before projection.
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        use_online_memory=False,
        attn_max_len=2,
        keep_max_len=2,
        causal=False,
        is_box_attn=False,
        online_compress=False,
        compress_kernel=(1, 1, 1),
        compress_stride=(1, 1, 1),
        sep_mem_rel_sp=False,
        drop_attn_rate=0.0,
        drop_qkv_rate=0.0,
        cfg=None,
        conv_q=0,
    ):
        super().__init__()
        self.mode = mode

        self.use_online_memory = use_online_memory
        self.causal = causal
        if use_online_memory:
            self.cached_k = []
            self.cached_v = []
            # We cache video names, so that we can perform
            # masking for memory that comes from the previous
            # video.
            self.cached_video_names = []
            self.attn_max_len = attn_max_len
            self.keep_max_len = keep_max_len
        self.conv_q = conv_q

        self.is_box_attn = is_box_attn
        self.online_compress = online_compress

        self.pool_first = pool_first
        self.drop_rate = drop_rate
        self.drop_attn_rate = drop_attn_rate
        self.drop_qkv_rate = drop_qkv_rate
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads

        self.scale = head_dim**-0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.compress_stride = compress_stride

        self.separate_qkv = cfg.MVIT.SEPARATE_QKV

        if pool_first or self.separate_qkv:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)
        if drop_attn_rate > 0.0:
            self.attn_drop = nn.Dropout(drop_attn_rate)
        if drop_qkv_rate > 0.0:
            self.qkv_drop = nn.Dropout(drop_qkv_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()

        self.shift_q = 0
        self.shift_kv = 0
        if self.is_box_attn:
            pass
        elif mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv" or mode == "conv_unshared":
            if pool_first:
                dim_conv = dim // num_heads if mode == "conv" else dim
            else:
                dim_conv = dim_out // num_heads if mode == "conv" else dim_out

            if self.causal:
                if stride_q:
                    assert stride_q[0] == 1
                    self.shift_q = padding_q[0] = (
                        kernel_q[0] - 1 if len(kernel_q) > 0 else None
                    )
                if stride_kv:
                    assert stride_kv[0] == 1
                    self.shift_kv = padding_kv[0] = (
                        kernel_kv[0] - 1 if len(kernel_kv) > 0 else None
                    )

            self.pool_q = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        if self.online_compress:
            compress_padding = [int(kv // 2) for kv in compress_kernel]
            self.compress_k = nn.Conv3d(
                dim,
                dim,
                compress_kernel,
                stride=compress_stride,
                padding=compress_padding,
                groups=dim,
                bias=False,
            )
            self.compress_v = nn.Conv3d(
                dim,
                dim,
                compress_kernel,
                stride=compress_stride,
                padding=compress_padding,
                groups=dim,
                bias=False,
            )
            self.norm_compress_k = norm_layer(dim) if len(compress_kernel) > 0 else None
            self.norm_compress_v = norm_layer(dim) if len(compress_kernel) > 0 else None

        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        if self.rel_pos_spatial:
            assert input_size[1] == input_size[2]
            q_size = (
                input_size[1] // stride_q[1] if len(stride_q) > 0 else input_size[1]
            )
            kv_size = (
                input_size[1] // stride_kv[1] if len(stride_kv) > 0 else input_size[1]
            )
            rel_sp_dim = 2 * max(q_size, kv_size) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            trunc_normal_(self.rel_pos_h, std=0.02)
            trunc_normal_(self.rel_pos_w, std=0.02)

            self.rel_pos_mem_h = None
            self.rel_pos_mem_w = None
            if self.use_online_memory:
                if self.online_compress:
                    mem_kv_size = kv_size // compress_stride[1]
                    mem_rel_sp_dim = 2 * max(q_size, mem_kv_size) - 1
                    self.rel_pos_mem_h = nn.Parameter(
                        torch.zeros(mem_rel_sp_dim, head_dim)
                    )
                    self.rel_pos_mem_w = nn.Parameter(
                        torch.zeros(mem_rel_sp_dim, head_dim)
                    )
                    trunc_normal_(self.rel_pos_mem_h, std=0.02)
                    trunc_normal_(self.rel_pos_mem_w, std=0.02)

        if self.rel_pos_temporal:
            self.rel_pos_t = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            if not cfg.TRAIN.CHECKPOINT_IN_INIT:
                trunc_normal_(self.rel_pos_t, std=0.02)

            self.rel_pos_mem_t = None
            if self.use_online_memory:
                if self.use_online_memory:
                    extend_size = (attn_max_len - 2) * input_size[0]
                    self.rel_pos_mem_t = nn.Parameter(
                        torch.zeros(2 * input_size[0] + extend_size - 1, head_dim)
                    )
                    trunc_normal_(self.rel_pos_mem_t, std=0.02)

        if self.causal:
            self.causal_mask = torch.zeros(input_size[0], input_size[0])
            for i in range(input_size[0]):
                for j in range(i + 1, input_size[0]):
                    self.causal_mask[i, j] = float("-inf")

        if self.conv_q > 0:
            self.conv_q_func = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    padding=padding_kv,
                    groups=head_dim,
                )
                if self.conv_q != 1
                else nn.Identity()
            )

    def forward(self, x, thw_shape, mem_selections=None, video_names=None):
        B, N, C = x.shape
        if self.mode == "conv_unshared":
            fold_dim = 1
        else:
            fold_dim = self.num_heads

        if self.pool_first:

            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            if self.separate_qkv:
                q = k = v = x
                q = self.q(q).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
                k = self.k(k).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
                v = self.v(v).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            else:
                qkv = self.qkv(x).reshape(B, N, 3, fold_dim, -1).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

        if not self.is_box_attn:
            q, q_shape = attention_pool(
                q,
                self.pool_q,
                thw_shape,
                has_cls_embed=self.has_cls_embed,
                norm=self.norm_q if hasattr(self, "norm_q") else None,
                shift_t=self.shift_q,
            )
            k, k_shape = attention_pool(
                k,
                self.pool_k,
                thw_shape,
                has_cls_embed=self.has_cls_embed,
                norm=self.norm_k if hasattr(self, "norm_k") else None,
                shift_t=self.shift_kv,
            )
            v, v_shape = attention_pool(
                v,
                self.pool_v,
                thw_shape,
                has_cls_embed=self.has_cls_embed,
                norm=self.norm_v if hasattr(self, "norm_v") else None,
                shift_t=self.shift_kv,
            )

        mem_k_shape = None
        mem_k_len = None
        if self.pool_first:
            if self.is_box_attn:
                q_N = k_N = v_N = 100
            else:
                q_N = (
                    numpy.prod(q_shape) + 1
                    if self.has_cls_embed
                    else numpy.prod(q_shape)
                )
                k_N = (
                    numpy.prod(k_shape) + 1
                    if self.has_cls_embed
                    else numpy.prod(k_shape)
                )
                v_N = (
                    numpy.prod(v_shape) + 1
                    if self.has_cls_embed
                    else numpy.prod(v_shape)
                )

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            q = self.q(q).reshape(B, q_N, self.num_heads, -1).permute(0, 2, 1, 3)

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)

            if self.use_online_memory:
                (
                    self.cached_k,
                    new_mem_selections,
                    new_cached_video_names,
                ) = mask_memory(
                    self.cached_k,
                    self.cached_video_names,
                    video_names,
                    mem_selections,
                )
                (
                    self.cached_v,
                    new_mem_selections,
                    new_cached_video_names,
                ) = mask_memory(
                    self.cached_v,
                    self.cached_video_names,
                    video_names,
                    mem_selections,
                )
                self.cached_video_names = new_cached_video_names
                mem_selections = new_mem_selections

                # Cache memory
                self.cached_k.append(k.detach())
                self.cached_v.append(v.detach())
                self.cached_video_names.append(video_names)
                if len(self.cached_k) > 1:
                    if self.online_compress:
                        # expect B, H, N, C.  B, N, C becomes B, 1, N, C.
                        self.cached_k[-2], mem_k_shape = attention_pool(
                            self.cached_k[-2],
                            self.compress_k,
                            k_shape,
                            has_cls_embed=self.has_cls_embed,
                            norm=self.norm_compress_k,
                        )

                        self.cached_v[-2], _ = attention_pool(
                            self.cached_v[-2],
                            self.compress_v,
                            v_shape,
                            has_cls_embed=self.has_cls_embed,
                            norm=self.norm_compress_v,
                        )

                    elif self.rel_pos_spatial or self.rel_pos_temporal:
                        if isinstance(k_shape[0], int):
                            mem_k_shape = k_shape.copy()
                        else:
                            mem_k_shape = [i.clone() for i in k_shape]

                bs = k.shape[0]
                k = [self.cached_k[:-1][i][:bs] for i in mem_selections] + [k]
                v = [self.cached_v[:-1][i][:bs] for i in mem_selections] + [v]

                if self.rel_pos_spatial or self.rel_pos_temporal:
                    if len(k) > 1:
                        mem_k_shape[0] *= len(k) - 1
                        mem_k_len = len(k) - 1
                    else:
                        mem_k_shape = None
                        mem_k_len = None

                if len(self.cached_k) > 1:
                    self.cached_k[-2] = self.cached_k[-2].detach()
                    self.cached_v[-2] = self.cached_v[-2].detach()

                # We keep at most (self.keep_max_len - 1) memory.
                if len(self.cached_k) == self.keep_max_len:
                    self.cached_k, self.cached_v = (
                        self.cached_k[1:],
                        self.cached_v[1:],
                    )
                    self.cached_video_names = self.cached_video_names[1:]
            if isinstance(k, (tuple, list)):
                k = torch.cat(k, dim=1)
            if isinstance(v, (tuple, list)):
                v = torch.cat(v, dim=1)
            k = self.k(k)
            v = self.v(v)
            k = k.reshape(
                B, -1, self.num_heads, k.shape[-1] // self.num_heads
            ).permute(0, 2, 1, 3)
            v = v.reshape(
                B, -1, self.num_heads, v.shape[-1] // self.num_heads
            ).permute(0, 2, 1, 3)

        if self.drop_qkv_rate > 0.0:
            q = self.qkv_drop(q)
            k = self.qkv_drop(k)
            v = self.qkv_drop(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h,
                self.rel_pos_w,
                mem_k_shape,
                self.rel_pos_mem_h,
                self.rel_pos_mem_w,
                mem_k_len,
            )

        if self.rel_pos_temporal:
            attn = cal_rel_pos_temporal(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_t,
                mem_k_shape,
                self.rel_pos_mem_t,
                self.compress_stride[0] if self.online_compress else 1,
                mem_k_len,
            )

        if self.causal:
            mask = (
                self.causal_mask.to(attn.device)
                .reshape(q_shape[0], 1, 1, k_shape[0], 1, 1)
                .repeat(1, q_shape[1], q_shape[2], 1, k_shape[1], k_shape[2])
            )
            mask = mask.reshape(1, 1, numpy.prod(q_shape), numpy.prod(k_shape))

            sp_idx = 1 if self.has_cls_embed else 0
            attn[:, :, sp_idx:, -mask.shape[-1] :] += mask

            if self.has_cls_embed:
                # Can't attend to cls of the current clip.
                attn[:, :, :, -mask.shape[-1] - 1] = float("-inf")
        attn = attn.softmax(dim=-1)
        if self.drop_attn_rate > 0.0:
            attn = self.attn_drop(attn)

        N = q.shape[2]

        if self.conv_q > 0:
            # print(q_shape, q.shape)
            if self.is_box_attn:
                x = (attn @ v + q).transpose(1, 2).reshape(-1, N, self.dim_out)
            else:
                conv_q = get_conv_q(
                    q,
                    q_shape,
                    self.conv_q_func,
                    has_cls_embed=self.has_cls_embed,
                )
                x = (attn @ v + conv_q).transpose(1, 2).reshape(-1, N, self.dim_out)
        else:
            x = (attn @ v).transpose(1, 2).reshape(-1, N, self.dim_out)

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        return x, (None if self.is_box_attn else q_shape)


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        input_size,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        use_online_memory=False,
        attn_max_len=2,
        keep_max_len=2,
        causal=False,
        is_box_attn=False,
        online_compress=False,
        compress_kernel=(1, 1, 1),
        compress_stride=(1, 1, 1),
        sep_mem_rel_sp=False,
        drop_attn_rate=0.0,
        drop_qkv_rate=0.0,
        cfg=None,
        conv_q=0,
        dim_mul_in_att=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.dim_mul_in_att = dim_mul_in_att

        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]

        self.is_box_attn = is_box_attn
        att_dim = dim_out if dim_mul_in_att else dim
        self.attn = MultiScaleAttention(
            dim,
            att_dim,
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_temporal=rel_pos_temporal,
            use_online_memory=use_online_memory,
            attn_max_len=attn_max_len,
            keep_max_len=keep_max_len,
            causal=causal,
            is_box_attn=is_box_attn,
            online_compress=online_compress,
            compress_kernel=compress_kernel,
            compress_stride=compress_stride,
            sep_mem_rel_sp=sep_mem_rel_sp,
            drop_attn_rate=drop_attn_rate,
            drop_qkv_rate=drop_qkv_rate,
            cfg=cfg,
            conv_q=conv_q,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        if causal:
            assert (not kernel_skip) or kernel_skip[0] == 1
        self.pool_skip = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(kernel_skip) > 0 and not is_box_attn
            else None
        )

    def forward(self, x, thw_shape, mem_selections=None, video_names=None):
        x_norm = self.norm1(x)
        x_block, thw_shape_new = self.attn(
            x_norm, thw_shape, mem_selections, video_names
        )
        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)

        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
        )
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new


def mask_memory(cached_mem, cached_video_names, cur_video_names, mem_selections):
    """
    Masking memory so that one video cannot attend to memory of a different video.
    """
    assert len(cached_mem) == len(cached_video_names)
    kept_ts = []
    for t, (cached_mem_t, cached_video_names_t) in enumerate(
        zip(cached_mem, cached_video_names)
    ):
        assert len(cached_video_names_t) == cached_mem_t.shape[0]
        assert len(cached_video_names_t) >= len(cur_video_names)
        keep = True
        for i in range(len(cur_video_names)):
            if cached_video_names_t[i] != cur_video_names[i]:
                cached_mem_t[i] = 0.0
                keep = False
        if keep:
            kept_ts.append(t)

    if len(cached_mem) > 0 and cached_mem[0].shape[0] == 1:
        old_to_new_t = {t: i for i, t in enumerate(kept_ts)}
        return (
            [cached_mem[t] for t in kept_ts],
            [old_to_new_t[old_t] for old_t in mem_selections if old_t in old_to_new_t],
            [cached_video_names[t] for t in kept_ts],
        )
    return cached_mem, mem_selections, cached_video_names
