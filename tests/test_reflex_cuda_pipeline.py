"""Real CUDA/JIT tests. Run on the deployment GPU before selecting Triton."""
import importlib.util
import pytest
import torch
from helper.fast_lk_reflex import FastLKReflex
from helper.tree_verification import PackedTree, VerifiedPath, trace_verified_path

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
                                reason="real CUDA + Triton required (CPU equations are separate tests)")


@pytest.mark.parametrize("batch", [1, 3, 64])
@pytest.mark.parametrize("contexts", [1, 4, 7, 8])
@pytest.mark.parametrize("dim", [4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("greedy", [False, True])
def test_real_multicontext_topk_and_visited_feedback(batch, contexts, dim, dtype, greedy):
    from helper import fast_lk_reflex_kernels as kernels
    torch.manual_seed(901)
    vocab, hidden_size, k = 16003, 32, 8
    engines = [FastLKReflex(feature_dim=dim, backend=backend, feedback_scope="visited_path", weight_decay=.1)
               for backend in ("torch", "triton")]
    for engine in engines:
        engine.start(batch, vocab, hidden_size, "cuda", max_contexts=contexts, max_path_length=3)
    reference, candidate = engines
    reference.state.normal_(std=.1)
    candidate.state.copy_(reference.state)
    raw = torch.randn(batch, contexts, vocab * 2, device="cuda", dtype=dtype)[..., ::2]
    hidden = torch.randn(batch, contexts, hidden_size, device="cuda", dtype=dtype)
    mapping = torch.arange(vocab, device="cuda") + 3
    q = reference.correct(raw, hidden)
    actual_q = candidate.correct(raw, hidden)
    torch.testing.assert_close(actual_q, q, rtol=2e-4, atol=2e-7)
    psi = candidate._feature(hidden)
    values, ids, norm = kernels.propose(raw, psi, candidate.state, k, candidate._proposal_workspace)
    expected = q.topk(k)
    torch.testing.assert_close(values, expected.values, rtol=2e-4, atol=2e-7)
    # Different IDs are allowed only within near-equal scores; never a worse
    # candidate outside numerical tolerance. Torch's tie ordering is unspecified.
    torch.testing.assert_close(q.gather(-1, ids), expected.values, rtol=2e-4, atol=2e-7)
    for engine in engines:
        engine.propose(raw[:, :1], hidden[:, :1], k, mapping, root=True)
        if contexts > 1:
            engine.propose(raw[:, 1:], hidden[:, 1:], k, mapping)
    indices = torch.tensor([0, 2, -1], device="cuda").expand(batch, -1)
    ctx = torch.tensor([0, contexts - 1 if contexts > 1 else -1, -1], device="cuda").expand(batch, -1).clone()
    if batch > 1:
        ctx[:batch // 2, 1] = -1  # different eligible path lengths
    path = VerifiedPath(indices, indices, ctx, (ctx >= 0).sum(1))
    teacher = torch.randn(batch, 4, vocab + 9, device="cuda").softmax(-1)
    if greedy:
        teacher = teacher.argmax(-1)
        teacher[:, 0], teacher[:, 2] = vocab + 8, mapping[7]  # outside/inside compact
    reference.update_visited(teacher, mapping, path, greedy=greedy)
    candidate.update_visited(teacher, mapping, path, greedy=greedy)
    torch.testing.assert_close(candidate.state, reference.state, rtol=3e-4, atol=3e-6)
    if batch > 1:
        keep = torch.arange(batch, device="cuda")[1:]
        for engine in engines:
            engine.remove_finished([0])
            engine.propose(raw[keep, :1], hidden[keep, :1], k, mapping, root=True)
            if contexts > 1:
                engine.propose(raw[keep, 1:], hidden[keep, 1:], k, mapping)
        smaller_path = VerifiedPath(indices[1:], indices[1:], ctx[1:], path.lengths[1:])
        for engine in engines:
            engine.update_visited(teacher[1:], mapping, smaller_path, greedy=greedy)
        torch.testing.assert_close(candidate.state, reference.state, rtol=3e-4, atol=3e-6)
    tree = PackedTree(torch.tensor([-1, 0, 0, 1, 1, 3], device="cuda").expand(batch, -1).contiguous(),
        torch.tensor([-1, 3, 3, 4, 5, 6], device="cuda").expand(batch, -1).contiguous(),
        torch.tensor([0, 1, 2, 3, -1, -1], device="cuda").expand(batch, -1).contiguous(), 3)
    samples = torch.tensor([3, 4, 9, 6, 8, 2], device="cuda").expand(batch, -1).contiguous()
    expected_path = trace_verified_path(tree, samples, 2)
    actual_path = trace_verified_path(tree, samples, 2, kernels=kernels)
    for name in ("tokens", "packed_indices", "feedback_contexts", "lengths"):
        assert torch.equal(getattr(actual_path, name), getattr(expected_path, name))
    actual_mask = tree.attention_mask(13, torch.bfloat16, kernels=kernels)
    expected_mask = tree.attention_mask(13, torch.bfloat16)
    torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)
