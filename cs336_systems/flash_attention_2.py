
import triton
import triton.language as tl

import math
import torch


TILE_SIZE = 16

def recompute_bwd(dO, Q, K, V, O, L, is_causal):
    """Pytorch impl of recompute backward of attention"""
    d = Q.shape[-1]
    D = torch.sum(O * dO, dim=-1, keepdim=True)  # (b, s, 1)
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)  # (b, s, s)
    # apply the same masking here, since we are not tiling, this could be simplified
    # as a lower triangle matrix
    if is_causal:
        q_indice = torch.arange(0, Q.shape[1], device=Q.device)
        k_indice = torch.arange(0, K.shape[1], device=K.device)
        mask = q_indice[:, None] < k_indice[None, :]  # (s, s)
        S = torch.where(mask, -1e6, S)
    P = torch.exp(S - L[:, :, None])  # (b, s, s)
    dV = torch.matmul(P.transpose(-2, -1), dO)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    dS = P * (dP - D)
    dQ = torch.matmul(dS, K) / math.sqrt(d)
    dK = torch.matmul(dS.transpose(-2, -1), Q) / math.sqrt(d)

    return dQ, dK, dV

compiled_recompute_bwd = torch.compile(recompute_bwd)

class FlashAttentionPyTorch(torch.autograd.Function):
    """PyTorch implementation of flash attention 2"""

    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        # input shape could be multiple dimensions, reshape
        q_shape = q.shape
        # q, k, v are 2D tensor right now
        nq, d = q.shape[-2], q.shape[-1]
        nk  = k.shape[-2]
        num_q_tiles = nq // TILE_SIZE
        num_kv_tiles = nk // TILE_SIZE
        # initialize the global o, l to be saved in ctx
        o = torch.empty_like(q, dtype=q.dtype, device=q.device)
        l = torch.empty(q_shape[:-1], dtype=q.dtype, device=q.device).unsqueeze(-1)
        # holding the underscore as in the original paper
        for i in range(0, num_q_tiles):
            # tensor slicing to mimic the loading from HBM to SRAM
            qi = q[:, i*TILE_SIZE:(i+1)*TILE_SIZE, :]  # (B, TILE_SIZE, D)
            # initialize the local o, l, m
            oi = torch.zeros_like(qi, dtype=qi.dtype, device=qi.device)
            li = torch.zeros(qi.shape[:-1], dtype=qi.dtype, device=qi.device).unsqueeze(-1)
            mi = torch.full(qi.shape[:-1], -float("inf"), dtype=qi.dtype, device=qi.device).unsqueeze(-1)
            # second loop to iterate each k,v tile
            for j in range(0, num_kv_tiles):
                kj = k[:, j*TILE_SIZE:(j+1)*TILE_SIZE, :]  # (B, TILE_SIZE, D)
                vj = v[:, j*TILE_SIZE:(j+1)*TILE_SIZE, :]  # (B, TILE_SIZE, D)
                # compute the attention scores
                sij = torch.matmul(qi, kj.transpose(-2, -1)) / math.sqrt(d)  # (B, TILE_SIZE, TILE_SIZE)
                mij = torch.concat([mi, sij], dim=-1).amax(dim=-1, keepdim=True)  # (B, TILE_SIZE, 1)
                pij = torch.exp(sij - mij)
                li = torch.exp(mi - mij) * li + torch.sum(pij, dim=-1, keepdim=True)
                # broadcasting is similar 
                oi = torch.exp(mi - mij) * oi + torch.matmul(pij, vj)
                # overwrite the mi to the current mij
                mi = mij
            oi = li ** -1 * oi
            li = mi + torch.log(li)
            o[:, i*TILE_SIZE:(i+1)*TILE_SIZE, :] = oi
            l[:, i*TILE_SIZE:(i+1)*TILE_SIZE, :] = li
        
        o = o.view(q_shape)
        l = l.view(q_shape[:-1])
        ctx.save_for_backward(q, k, v, o, l)

        return o

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = compiled_recompute_bwd(dO, Q, K, V, O, L, False)
        return dQ, dK, dV, None

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # in this kernel impl, each thread would process a single sample in the batch
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),  # we would loop through the k,v
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(  # v should be pretty similar to K
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # initialize buffers
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q_indice = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
    # loop k,v tiles
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        ki = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        vi = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        si = tl.dot(q, tl.trans(ki)) * scale
        if is_causal:  # is there a better way to implement this?
            k_indice = tl.arange(0, K_TILE_SIZE) + i * K_TILE_SIZE
            mask = q_indice[:, None] < k_indice[None, :]
            si = tl.where(mask, -1e6, si)
        mi = tl.maximum(m, tl.max(si, axis=1))  # new maximum per row
        pi = tl.exp(si - mi[:, None])
        l = tl.exp(m - mi) * l + tl.sum(pi, axis=1)
        pi = pi.to(vi.dtype)  # cast p to the same dtype of v
        o = tl.exp(m - mi)[:, None] * o + tl.dot(pi, vi)
        m = mi
    
        # advance the block
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # final normalization
    o = o / l[:, None]
    l = m + tl.log(l)

    # o is computed in float32, needs to convert it back to Q/O dtype
    tl.store(O_block_ptr, o.to(q.dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal):
        
        # initialize the buffer for the fa2 kernel
        O = torch.empty_like(Q, dtype=Q.dtype, device=Q.device)
        L = torch.empty(Q.shape[:-1], dtype=torch.float32, device=Q.device)

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        bsz, N_QUERIES, D = Q.shape[0], Q.shape[1], Q.shape[2]
        N_KEYS = K.shape[1]
        scale = 1 / math.sqrt(D)

        # the launch grid is (Tq, bsz), so that each thread would only process a tile
        # of query from a single batch sample
        flash_fwd_kernel[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), bsz)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE
        ctx.is_causal = is_causal
        # assume a (bsz, s, d) shape, need to change if the input is with head dimension
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = compiled_recompute_bwd(dO, Q, K, V, O, L, is_causal)
        return dQ, dK, dV, None