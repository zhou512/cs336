import torch
import torch.nn as nn
import triton
import triton.language as tl
@triton.jit
def flashattention_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oa, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    #Q_TILE_SIZE = Bq, K_TILE_SIZE = Bk
    # Q,O:(BATCH_SIZE, N_QUERIES, D), K,V:(BATCH_SIZE, N_KEYS, D), L:(BATCH_SIZE, N_QUERIES)
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
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
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
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
        strides=(stride_oa, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,),
    )
    if is_causal:
        # 只遍历到当前 Query 块的最大行号即可
        hi = tl.minimum(N_KEYS, (query_tile_index + 1) * Q_TILE_SIZE)
    else:
        hi = N_KEYS
        #行号
    offs_m = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    Tk = (N_KEYS + K_TILE_SIZE - 1) // (K_TILE_SIZE)
    q_i = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option='zero')
    o_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    m_i = tl.full([Q_TILE_SIZE], float('-inf'), dtype=tl.float32)
    m_ij = m_i
    for j in range(0, hi, K_TILE_SIZE):
        k = tl.load(K_block_ptr, boundary_check=(0,1), padding_option='zero') #(K_TILE_SIZE,D)
        v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option='zero')
        s_ij = tl.dot(q_i, tl.trans(k)) * scale #(Q_TILE_SIZE,K_TILE_SIZE)
        offs_d = j + tl.arange(0, K_TILE_SIZE)
        valid = offs_d[None, :] < N_KEYS

        if is_causal:
            valid = valid & (offs_m[:, None] >= offs_d[None, :])

        s_ij = tl.where(valid, s_ij, -1e6)
        rms_ij = tl.max(s_ij, axis=1)
        m_ij_prev = m_ij
        m_ij_new = tl.maximum(m_ij_prev, rms_ij)
        m_ij = m_ij_new
        p_ij = tl.exp(s_ij - m_ij_new[:, None]) #(Q_TILE_SIZE,K_TILE_SIZE)
        m_diff = tl.exp(m_ij_prev - m_ij_new)
        l_i = m_diff * l_i + tl.sum(p_ij, axis=1)
        o_i = m_diff[:, None] * o_i + tl.dot(p_ij.to(v.dtype), v)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    o_i = o_i / l_i[:, None]
    L_i = m_ij + tl.log(l_i)
    tl.store(O_block_ptr, o_i.to(O_block_ptr.dtype.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, L_i.to(L_block_ptr.dtype.element_ty), boundary_check=(0,))

@triton.jit
def flashattention_bwdkv(
    Q_ptr, K_ptr, dK_ptr, V_ptr, dV_ptr, dO_ptr, D_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_vb, stride_vk, stride_vd,
    stride_dvb, stride_dvk, stride_dvd,
    stride_dob, stride_doa, stride_dod,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D_MODEL: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Q,O,dO:(BATCH_SIZE, N_QUERIES, D_MODEL), K,V:(BATCH_SIZE, N_KEYS, D_MODEL), L:(BATCH_SIZE, N_QUERIES)
    key_tile_index = tl.program_id(0)
    if is_causal:
        start_q_block = (key_tile_index * K_TILE_SIZE) // Q_TILE_SIZE
        start_q = start_q_block * Q_TILE_SIZE
        start_q = tl.minimum(start_q, N_QUERIES) # 防止越界
    else:
        start_q = 0
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(start_q, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
       dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_doa, stride_dod),
        offsets=(start_q, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1,0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape = (N_QUERIES,),
        strides=(stride_dq,),
        offsets=(start_q,),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides=(stride_lq,),
        offsets=(start_q,),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,),
    )
    K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
    V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')
    dK_ij = tl.zeros([K_TILE_SIZE, D_MODEL], dtype=tl.float32)
    dV_ij = tl.zeros([K_TILE_SIZE, D_MODEL], dtype=tl.float32)

    offs_k = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
    for i in range(start_q, N_QUERIES, Q_TILE_SIZE):
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option='zero')
        Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
        dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option='zero')
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option='zero')
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale #(Bq, Bk)
        offs_q = i + tl.arange(0, Q_TILE_SIZE)
        valid = (offs_q[:, None] < N_QUERIES) & (offs_k[None, :] < N_KEYS)

        if is_causal:
            valid = valid & (offs_q[:, None] >= offs_k[None, :])

        S_ij = tl.where(valid, S_ij, -1e6)
        P_ij = tl.exp(S_ij - L_i[:, None]) #(Bq, Bk) - (Bq) = (Bq, Bk)
        dV_ij += tl.dot(tl.trans(P_ij), dO_i) #(Bk, d)
        dP_ij = tl.dot(dO_i, tl.trans(V_j)) #(Bq, Bk)
        dS_ij = P_ij * (dP_ij - D_i[:, None]) #(Bq, Bk)
        dK_ij += tl.dot(tl.trans(dS_ij), Q_i) * scale
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
    tl.store(dK_block_ptr, dK_ij.to(dK_block_ptr.dtype.element_ty), boundary_check=(0,1))
    tl.store(dV_block_ptr, dV_ij.to(dV_block_ptr.dtype.element_ty), boundary_check=(0,1))

@triton.jit
def flashattention_bwdq(
    Q_ptr, dQ_ptr, K_ptr, V_ptr, dO_ptr, D_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_dqb, stride_dqq, stride_dqd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doa, stride_dod,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D_MODEL: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Q,O,dO:(BATCH_SIZE, N_QUERIES, D_MODEL), K,V:(BATCH_SIZE, N_KEYS, D_MODEL), L:(BATCH_SIZE, N_QUERIES)
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_doa, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1,0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape = (N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,),
    )
    L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option='zero')
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option='zero')
    D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option='zero')
    dQ_ij = tl.zeros([Q_TILE_SIZE, D_MODEL], dtype=tl.float32)
    if is_causal:
        # 只遍历到当前 Query 块的最大行号即可
        hi = tl.minimum(N_KEYS, (query_tile_index + 1) * Q_TILE_SIZE)
    else:
        hi = N_KEYS
        #行号
    offs_m = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    for j in range(0, hi, K_TILE_SIZE):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale

        offs_d = j + tl.arange(0, K_TILE_SIZE)
        valid = offs_d[None, :] < N_KEYS

        if is_causal:
            valid = valid & (offs_m[:, None] >= offs_d[None, :])

        S_ij = tl.where(valid, S_ij, -1e6)

        P_ij = tl.exp(S_ij - L_i[:, None])
        dP_ij = tl.dot(dO_i, tl.trans(V_j))
        dS_ij = P_ij * (dP_ij - D_i[:, None])
        dQ_ij += tl.dot(dS_ij, K_j) * scale
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    tl.store(dQ_block_ptr, dQ_ij.to(dQ_block_ptr.dtype.element_ty), boundary_check=(0,1))

class FlashAttention_pt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal=False):
        bq, bk = 32, 32
        #q.shape = (b, nq, d), k.shape = (b, nk, d) = v.shape
        #block.shape = (b, bq, bk)
        b, d = q.shape[0], q.shape[2]
        nq, nk = q.shape[1], k.shape[1]
        tq = (nq + bq - 1) // bq
        tk = (nk + bk - 1) // bk
        scale = d** -0.5
        #dividing blocks
        q_blocks = [q[:, i * bq: min((i + 1) * bq, nq), :] for i in range(tq)]
        k_blocks = [k[:, i * bk: min((i + 1) * bk, nk), :] for i in range(tk)]
        v_blocks = [v[:, i * bk: min((i + 1) * bk, nk), :] for i in range(tk)]

        O = torch.zeros_like(q)
        L = torch.zeros((b, nq), dtype=torch.float32, device=q.device)

        for i in range(tq):
            q_i = q_blocks[i] #(b, bq ,d)
            bq_curr = q_i.shape[1]
            o_i = torch.zeros_like(q_i)
            l_i = torch.zeros((b, bq_curr), device=q.device, dtype=torch.float32)
            m_i = torch.full((b, bq_curr), float('-inf'), dtype=torch.float32, device=q.device)
            m_ij, l_ij, o_ij = m_i, l_i, o_i
            for j in range(tk):
                k_j = k_blocks[j] #(b, bk, d)
                v_j = v_blocks[j]
                s_ij = q_i @ k_j.transpose(-1, -2) * scale #(b, bq, bk)
                if is_causal:
                    # 跳过右上角的块 (完全不需要算)
                    q_start = i * bq
                    q_end = q_start + q_i.shape[1]   # exclusive
                    k_start = j * bk
                    k_end = j * bk + k_j.shape[1]
                    if k_start >= q_end:
                        continue
                    # 处理对角线上的块 (需要加 Mask)
                    if k_end > q_start:
                        # 生成 Mask
                        bq_curr = q_i.shape[1]
                        bk_curr = k_j.shape[1]
                        row_idx = torch.arange(q_start, q_start + bq_curr, device=q.device)[:, None]
                        col_idx = torch.arange(k_start, k_start + bk_curr, device=q.device)[None, :]
                        mask = (row_idx >= col_idx) #(b, bq, bk)
                        s_ij = s_ij.masked_fill(~mask, float('-inf'))
                #在sij的每一行的bk个元素中，找到一个最大的元素
                rms_ij = s_ij.max(dim=-1, keepdim=False).values #(b, bq)
                m_ij_prev = m_ij
                m_ij = torch.maximum(m_ij_prev, rms_ij)  #(b, bq)
                m_ij = m_ij[..., None] # (b, bq, 1)
                p_ij = torch.exp(s_ij - m_ij) #(b, bq, bk)
                m_ij.squeeze_(dim=-1)
                m_diff = torch.exp(m_ij_prev - m_ij) #(b, bq)
                l_ij = m_diff * l_ij + p_ij.sum(dim=-1, keepdim=False)
                #torch.diag(m_diff).shape = (bq,bq)
                o_ij = m_diff[..., None] * o_ij + p_ij @ v_j #(b, bq, d)
            o_i = o_ij / l_ij[..., None]
            L_i = m_ij + torch.log(l_ij)
            O[:, i * bq : i * bq + bq_curr, :] = o_i
            L[:, i * bq : i * bq + bq_curr] = L_i
        ctx.save_for_backward(q, k, v, O ,L)
        ctx.is_causal = is_causal
        return O
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        # Q, O, D (b, nq, d) K,V(b, nk, d), L(b,nq) 
        d = Q.shape[-1]
        scale = d ** -0.5
        D = (O * dO).sum(dim=-1, keepdim=True)
        S = torch.matmul(Q, K.transpose(-1, -2)) * scale
        if ctx.is_causal:                                      # ★ 关键修复
            nq, nk = Q.shape[1], K.shape[1]
            mask = torch.tril(torch.ones(nq, nk, device=Q.device, dtype=torch.bool))
            S = S.masked_fill(~mask, float('-inf'))
        # S (b, nq, nk)
        P = torch.exp(S - L[:,:,None])
        # P(b,nq, nk)
        dV = torch.matmul(P.transpose(-1, -2), dO)
        dP = torch.matmul(dO, V.transpose(-1, -2))
        dS = P * (dP - D)
        dQ = torch.matmul(dS, K) * scale
        dK = torch.matmul(dS.transpose(-1, -2), Q) * scale
        return dQ, dK, dV, None

class FlashAttention_tr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, Q_TILE_SIZE=None, K_TILE_SIZE=None):
        D = Q.shape[-1]

        if Q_TILE_SIZE is None or K_TILE_SIZE is None:
            if D >= 128:
                Q_TILE_SIZE = 16
                K_TILE_SIZE = 32
            elif D >= 64:
                Q_TILE_SIZE = 32
                K_TILE_SIZE = 32
            else:
                Q_TILE_SIZE = 64
                K_TILE_SIZE = 64


        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE
        ctx.is_causal = is_causal
        N_QUERIES, N_KEYS = Q.shape[1], K.shape[1]
        BATCH_SIZE = Q.shape[0]
        O_shape = Q.shape
        O = torch.empty(O_shape, device=Q.device, dtype=Q.dtype)
        L = torch.empty((BATCH_SIZE, N_QUERIES), device=Q.device)
        scale = D ** -0.5
        flashattention_fwd[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), BATCH_SIZE,)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS, scale,
            D, Q_TILE_SIZE, K_TILE_SIZE,
            is_causal
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O
    
    """ @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dtype = Q.dtype
        # Q, O, D (b, nq, d) K,V(b, nk, d), L(b,nq) 
        L = L.float()
        d = Q.shape[-1]
        scale = d ** -0.5
        D = (O * dO).sum(dim=-1, keepdim=True)
        S = torch.matmul(Q, K.transpose(-1, -2)) * scale
        if ctx.is_causal:                                      # ★ 关键修复
            nq, nk = Q.shape[1], K.shape[1]
            mask = torch.tril(torch.ones(nq, nk, device=Q.device, dtype=torch.bool))
            S = S.masked_fill(~mask, float('-inf'))
        # S (b, nq, nk)
        P = torch.exp(S - L[:,:,None])
        # P(b,nq, nk)
        dV = torch.matmul(P.transpose(-1, -2), dO)
        dP = torch.matmul(dO, V.transpose(-1, -2))
        dS = P * (dP - D)
        dQ = torch.matmul(dS, K) * scale
        dK = torch.matmul(dS.transpose(-1, -2), Q) * scale
        return dQ.to(dtype), dK.to(dtype), dV.to(dtype), None, None, None """

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dtype = Q.dtype
        Q, K, V, O, dO = Q.float(), K.float(), V.float(), O.float(), dO.float()
        L = L.float()
        N_QUERIES, N_KEYS, D_MODEL = Q.shape[1], K.shape[1], Q.shape[-1]
        scale = D_MODEL ** -0.5
        D = torch.sum(O * dO, dim=-1, keepdim=False)
        BATCH_SIZE = Q.shape[0]
        dQ = torch.empty([BATCH_SIZE, N_QUERIES, D_MODEL], dtype=dtype, device=Q.device)
        dK = torch.empty([BATCH_SIZE, N_KEYS, D_MODEL], dtype=dtype, device=K.device)
        dV = torch.empty([BATCH_SIZE, N_KEYS, D_MODEL], dtype=dtype, device=V.device)
        flashattention_bwdq[(triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), BATCH_SIZE,)](
            Q, dQ, K, V, dO, D, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS, scale,
            D_MODEL, ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE,
            ctx.is_causal
        )
        flashattention_bwdkv[(triton.cdiv(N_KEYS, ctx.K_TILE_SIZE), BATCH_SIZE,)](
            Q, K, dK, V, dV, dO, D, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS, scale,
            D_MODEL, ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE,
            ctx.is_causal
        )
        return dQ.to(dtype), dK.to(dtype), dV.to(dtype), None, None, None


