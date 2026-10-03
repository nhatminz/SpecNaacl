"""Lazy-loaded CUDA/Triton kernels for FP32 trajectory-local LK Reflex.

No model forward, autograd, RNG, target softmax, optimizer or host synchronization.
Reductions are tiled to support both compact and full Qwen vocabularies.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _feature_kernel(H, R, PSI, HS0, HS1,
                    HS2, CONTEXTS: tl.constexpr,
                    HIDDEN: tl.constexpr, DIM: tl.constexpr,
                    BH: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BH)
    d = tl.arange(0, BD)
    hidden = tl.load(H + (row // CONTEXTS) * HS0 + (row % CONTEXTS) * HS1 + h * HS2,
                     h < HIDDEN, other=0).to(tl.float32)
    projection = tl.load(R + h[:, None] * DIM + d[None, :],
                         (h[:, None] < HIDDEN) & (d[None, :] < DIM), other=0)
    projected = tl.sum(hidden[:, None] * projection, axis=0)
    norm = tl.maximum(tl.sqrt(tl.sum(projected * projected, axis=0)), 1.0e-6)
    tl.store(PSI + row * DIM + d, tl.div_rn(projected, norm), d < DIM)


@triton.jit
def _correction_kernel(Z, PSI, A, OUT,
                       ZS0, ZS1, ZS2,
                       VOCAB: tl.constexpr, DIM: tl.constexpr, CONTEXTS: tl.constexpr,
                       BV: tl.constexpr, BD: tl.constexpr):
    tile, context, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch = batch.to(tl.int64)
    v = tile * BV + tl.arange(0, BV)
    d = tl.arange(0, BD)
    a = tl.load(A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :],
                (v[:, None] < VOCAB) & (d[None, :] < DIM), other=0)
    psi = tl.load(PSI + (batch * CONTEXTS + context) * DIM + d, d < DIM, other=0)
    z = tl.load(Z + batch * ZS0 + context * ZS1 + v * ZS2, v < VOCAB, other=0).to(tl.float32)
    corrected = z + tl.sum(a * psi[None, :], axis=1)
    tl.store(OUT + (batch * CONTEXTS + context) * VOCAB + v, corrected, v < VOCAB)


@triton.jit
def _teacher_values(T, MAP, batch, v, VOCAB: tl.constexpr,
                    TS0, TS1, MS0,
                    GREEDY: tl.constexpr):
    ids = tl.load(MAP + v * MS0, v < VOCAB, other=0)
    if GREEDY:
        token = tl.load(T + batch * TS0)
        return ((ids == token) & (v < VOCAB)).to(tl.float32)
    return tl.load(T + batch * TS0 + ids * TS1, v < VOCAB, other=0).to(tl.float32)


@triton.jit
def _teacher_mass_kernel(T, MAP, MASS,
                         VOCAB: tl.constexpr, TS0, TS1,
                         MS0, GREEDY: tl.constexpr,
                         TILES: tl.constexpr, BV: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1)
    batch = batch.to(tl.int64)
    v = tile * BV + tl.arange(0, BV)
    p = _teacher_values(T, MAP, batch, v, VOCAB, TS0, TS1, MS0, GREEDY)
    tl.store(MASS + batch * TILES + tile, tl.sum(p, axis=0))


@triton.jit
def _lk_stats_kernel(Q, T, MAP, MASS, STATS,
                     QS0, QS1,
                     VOCAB: tl.constexpr, TS0, TS1,
                     MS0, GREEDY: tl.constexpr, EPS: tl.constexpr,
                     TILES: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1)
    batch = batch.to(tl.int64)
    t = tl.arange(0, BT)
    denominator = tl.sum(tl.load(MASS + batch * TILES + t, t < TILES, other=0), axis=0) + EPS
    v = tile * BV + tl.arange(0, BV)
    p = _teacher_values(T, MAP, batch, v, VOCAB, TS0, TS1, MS0, GREEDY)
    p = tl.div_rn(p, denominator)
    q = tl.load(Q + batch * QS0 + v * QS1, v < VOCAB, other=0).to(tl.float32)
    alpha = tl.sum(tl.minimum(p, q), axis=0)
    selected = tl.sum(tl.where(q < p, q, 0.0), axis=0)
    tl.store(STATS + (batch * TILES + tile) * 2, alpha)
    tl.store(STATS + (batch * TILES + tile) * 2 + 1, selected)


@triton.jit
def _state_update_kernel(A, Q, PSI, T, MAP, MASS, STATS, ALPHA,
                         QS0, QS1,
                         PS0, PS1,
                         VOCAB: tl.constexpr, DIM: tl.constexpr,
                         TS0, TS1, MS0,
                         GREEDY: tl.constexpr, EPS: tl.constexpr,
                         LR: tl.constexpr, DECAY: tl.constexpr,
                         TILES: tl.constexpr, BT: tl.constexpr,
                         BV: tl.constexpr, BD: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1)
    # Root probabilities are a strided view of [B, verification, full_vocab].
    # B*stride can exceed 2**31 on B200 even when the compact adapter is small.
    batch = batch.to(tl.int64)
    t = tl.arange(0, BT)
    mass = tl.sum(tl.load(MASS + batch * TILES + t, t < TILES, other=0), axis=0) + EPS
    alpha = tl.sum(tl.load(STATS + (batch * TILES + t) * 2, t < TILES, other=0), axis=0)
    selected = tl.sum(tl.load(STATS + (batch * TILES + t) * 2 + 1, t < TILES, other=0), axis=0)
    if tile == 0:
        tl.store(ALPHA + batch, alpha)
    if LR != 0.0 or DECAY != 1.0:
        v = tile * BV + tl.arange(0, BV)
        d = tl.arange(0, BD)
        p = _teacher_values(T, MAP, batch, v, VOCAB, TS0, TS1, MS0, GREEDY)
        p = tl.div_rn(p, mass)
        q = tl.load(Q + batch * QS0 + v * QS1, v < VOCAB, other=0).to(tl.float32)
        gradient = tl.div_rn(q * (selected - (q < p).to(tl.float32)), alpha + EPS)
        psi = tl.load(PSI + batch * PS0 + d * PS1, d < DIM, other=0).to(tl.float32)
        ptr = A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :]
        mask = (v[:, None] < VOCAB) & (d[None, :] < DIM)
        old = tl.load(ptr, mask, other=0)
        new = old * DECAY - LR * gradient[:, None] * psi[None, :]
        tl.store(ptr, new, mask)


def feature(hidden, projection):
    """Fuse FP32 conversion, seeded projection and feature normalization."""
    batch, contexts, hidden_size = hidden.shape
    dim = projection.shape[1]
    # Large unusual feature matrices are better left to cuBLAS; correction and
    # update remain fused. This is a shape-only choice, never a device sync.
    if triton.next_power_of_2(hidden_size) * triton.next_power_of_2(dim) > 32768:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return torch.nn.functional.normalize(hidden.float().matmul(projection), dim=-1, eps=1e-6)
    output = torch.empty((batch, contexts, dim), device=hidden.device, dtype=torch.float32)
    _feature_kernel[(batch * contexts,)](
        hidden, projection, output, *hidden.stride(), contexts, hidden_size, dim,
        triton.next_power_of_2(hidden_size), triton.next_power_of_2(dim),
        num_warps=8, enable_fp_fusion=False,
    )
    return output


def correct(logits, psi, state):
    """Fuse logit conversion, low-rank correction and addition; keep softmax."""
    batch, contexts, vocab = logits.shape
    dim = state.shape[-1]
    output = torch.empty((batch, contexts, vocab), device=logits.device, dtype=torch.float32)
    _correction_kernel[(triton.cdiv(vocab, 256), contexts, batch)](
        logits, psi, state, output, *logits.stride(), vocab, dim, contexts,
        256, triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False,
    )
    return output.softmax(dim=-1)


def update(state, root_q, root_psi, target, mapping, *, greedy, eps, learning_rate, decay):
    """Three tiled launches, no dense p/mask/gradient/outer-product temporaries."""
    batch, vocab, dim = state.shape
    tiles = triton.cdiv(vocab, 2048)
    mass = torch.empty((batch, tiles), device=state.device, dtype=torch.float32)
    stats = torch.empty((batch, tiles, 2), device=state.device, dtype=torch.float32)
    alpha = torch.empty((batch,), device=state.device, dtype=torch.float32)
    ts0 = target.stride(0)
    ts1 = 0 if greedy else target.stride(1)
    grid = (tiles, batch)
    _teacher_mass_kernel[grid](
        target, mapping, mass, vocab, ts0, ts1, mapping.stride(0), greedy, tiles, 2048,
        num_warps=4, enable_fp_fusion=False,
    )
    _lk_stats_kernel[grid](
        root_q, target, mapping, mass, stats, *root_q.stride(), vocab, ts0, ts1,
        mapping.stride(0), greedy, eps, tiles, triton.next_power_of_2(tiles), 2048,
        num_warps=4, enable_fp_fusion=False,
    )
    _state_update_kernel[(triton.cdiv(vocab, 128), batch)](
        state, root_q, root_psi, target, mapping, mass, stats, alpha,
        *root_q.stride(), *root_psi.stride(), vocab, dim, ts0, ts1, mapping.stride(0),
        greedy, eps, learning_rate, decay, tiles, triton.next_power_of_2(tiles),
        128, triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False,
    )
    return alpha
