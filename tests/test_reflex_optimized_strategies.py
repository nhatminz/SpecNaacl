"""Parity for opt-in parallel kernels and device-resident committed paths."""
import importlib.util

import pytest
import torch

from helper.tree_verification import VerifiedPath


def _legacy_pad(indices, tokens, past, width, eos):
    pad_budget, cursor = width - len(indices), past
    out_indices, out_tokens, pad = [], [], []
    last = width - 1
    for index, token in zip(indices, tokens):
        absolute = past + index
        if pad_budget > 0:
            while cursor != absolute and pad_budget:
                out_indices.append(cursor)
                out_tokens.append(eos)
                pad.append(True)
                cursor += 1
                pad_budget -= 1
            out_indices.append(absolute)
            out_tokens.append(token)
            pad.append(False)
            last = len(out_indices) - 1
            cursor += 1
        else:
            out_indices.append(absolute)
            out_tokens.append(token)
            pad.append(False)
            last = len(out_indices) - 1
    while pad_budget:
        out_indices.append(cursor)
        out_tokens.append(eos)
        pad.append(True)
        cursor += 1
        pad_budget -= 1
    return out_tokens, out_indices, pad, last


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_padded_path_matches_legacy_gap_and_tail_order(device):
    rows = [([0, 3, 7], [11, 12, 13]), ([0, 1], [21, 22]),
            ([0, 4, 8, 9], [31, 32, 33, 34]), ([0], [41])]
    capacity, past, eos = 5, 17, 99
    tokens = torch.full((len(rows), capacity), -1, device=device, dtype=torch.long)
    indices = torch.full_like(tokens, -1)
    for batch, (idx, tok) in enumerate(rows):
        indices[batch, :len(idx)] = torch.tensor(idx, device=device)
        tokens[batch, :len(tok)] = torch.tensor(tok, device=device)
    lengths = torch.tensor([len(idx) for idx, _ in rows], device=device)
    path = VerifiedPath(tokens, indices, indices, lengths)
    actual = path.padded_gpu(past, 4, eos)
    for batch, (idx, tok) in enumerate(rows):
        expected = _legacy_pad(idx, tok, past, 4, eos)
        for tensor, value in zip(actual, expected):
            assert tensor[batch].tolist() == (value if isinstance(value, list) else [value])
    if device == "cuda" and importlib.util.find_spec("triton") is not None:
        from helper import fast_lk_reflex_kernels as kernels
        fused = path.padded_gpu(past, 4, eos, kernels=kernels)
        for reference, candidate in zip(actual, fused):
            torch.testing.assert_close(candidate, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
                    reason="real CUDA/Triton required")
@pytest.mark.parametrize("contexts,dim,vocab", [(1, 4, 521), (4, 8, 521), (7, 8, 16003)])
def test_parallel_correction_sort_proposal_and_feedback(contexts, dim, vocab):
    from helper import fast_lk_reflex_kernels as kernels
    torch.manual_seed(317)
    batch, k, width = 3, 4, min(contexts, 3)
    raw = torch.randn(batch, contexts, vocab, device="cuda", dtype=torch.bfloat16)
    psi = torch.randn(batch, contexts, dim, device="cuda")
    state = torch.randn(batch, vocab, dim, device="cuda") * .03
    reference = kernels.correct_logits(raw, psi, state)
    for strategy in ("parallel", "tiled"):
        for bv, bc, warps in ((64, 1, 2), (128, 2, 4), (256, 4, 8)):
            corrected = kernels.correct_logits(raw, psi, state, strategy=strategy,
                block_vocab=bv, context_tile=bc, num_warps=warps)
            torch.testing.assert_close(corrected, reference, rtol=2e-5, atol=1e-6)
    old_values, old_ids, old_norm = kernels.propose(raw, psi, state, k)
    values, ids, norm = kernels.propose(raw, psi, state, k, strategy="sort")
    torch.testing.assert_close(values, old_values, rtol=2e-5, atol=1e-7)
    assert torch.equal(ids, old_ids)
    torch.testing.assert_close(norm, old_norm, rtol=2e-5, atol=1e-6)
    mapping = torch.arange(vocab, device="cuda") + 2
    target = torch.randn(batch, width, vocab + 9, device="cuda").softmax(-1)
    indices = torch.arange(width, device="cuda").expand(batch, -1).contiguous()
    path_contexts = indices.clone()
    if width > 1:
        path_contexts[0, -1] = -1
        indices[0, -1] = -1
    serial, parallel = state.clone(), state.clone()
    alpha_old = kernels.update_path(serial, raw, psi, old_norm, target, mapping,
        indices, path_contexts, greedy=False, eps=1e-8, learning_rate=.05, decay=.99)
    alpha_new = kernels.update_path(parallel, raw, psi, old_norm, target, mapping,
        indices, path_contexts, greedy=False, eps=1e-8, learning_rate=.05, decay=.99,
        strategy="parallel")
    torch.testing.assert_close(alpha_new, alpha_old, rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(parallel, serial, rtol=3e-4, atol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
                    reason="real CUDA/Triton required")
def test_reflex_update_stream_event_matches_single_stream():
    from helper.fast_lk_reflex import FastLKReflex
    from helper.tree_verification import VerifiedPath
    batch, vocab, dim = 3, 521, 8
    engines = [FastLKReflex(feature_dim=dim, backend="triton", feedback_scope="visited_path")
               for _ in range(2)]
    for engine in engines:
        engine.start(batch, vocab, 32, "cuda", max_contexts=2, max_path_length=2,
                     max_proposal_contexts=2, max_topk=4)
    raw = torch.randn(batch, 2, vocab, device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(batch, 2, 32, device="cuda", dtype=torch.bfloat16)
    mapping = torch.arange(vocab, device="cuda")
    teacher = torch.randn(batch, 2, vocab, device="cuda").softmax(-1)
    indices = torch.tensor([[0, 1], [0, 1], [0, -1]], device="cuda")
    path = VerifiedPath(indices, indices, indices, (indices >= 0).sum(1))
    for engine in engines:
        engine.propose(raw[:, :1], hidden[:, :1], 4, mapping, root=True)
        engine.propose(raw[:, 1:], hidden[:, 1:], 4, mapping)
    engines[0].update_visited(teacher, mapping, path)
    side = torch.cuda.Stream()
    ready, done = torch.cuda.Event(), torch.cuda.Event()
    ready.record(torch.cuda.current_stream())
    side.wait_event(ready)
    with torch.cuda.stream(side):
        engines[1].update_visited(teacher, mapping, path)
        done.record(side)
    # Independent accepted-token gather may run before the A dependency.
    gathered = teacher.gather(1, indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, vocab))
    torch.cuda.current_stream().wait_event(done)
    assert gathered.shape == teacher.shape
    torch.testing.assert_close(engines[1].state, engines[0].state, rtol=3e-4, atol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
                    reason="real CUDA/Triton required")
@pytest.mark.parametrize("proposal,correction", [("sort", "serial"), ("hybrid", "parallel"),
                                                  ("hybrid", "tiled"), ("torch", "serial")])
@pytest.mark.parametrize("amp", [False, True])
def test_opt_in_proposal_strategies_keep_root_feedback_equations(proposal, correction, amp):
    from helper.fast_lk_reflex import FastLKReflex
    torch.manual_seed(219)
    engines = [FastLKReflex(feature_dim=8, backend="triton", feedback_scope="root",
                            proposal_strategy=strategy, correction_strategy=correction)
               for strategy in ("fused", proposal)]
    for engine in engines:
        engine.start(3, 521, 32, "cuda", max_contexts=1, max_proposal_contexts=4,
                     max_path_length=1, max_topk=4)
    raw = torch.randn(3, 1, 521, device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(3, 1, 32, device="cuda", dtype=torch.bfloat16)
    mapping = torch.arange(521, device="cuda")
    teacher = torch.randn(3, 521, device="cuda").softmax(-1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        proposals = [engine.propose(raw, hidden, 4, mapping, root=True) for engine in engines]
    torch.testing.assert_close(proposals[1][0], proposals[0][0], rtol=3e-4, atol=2e-7)
    assert torch.equal(proposals[1][1], proposals[0][1])
    for engine in engines:
        engine.update_from_target_probs(teacher, mapping)
    torch.testing.assert_close(engines[1].state, engines[0].state, rtol=3e-4, atol=3e-6)
