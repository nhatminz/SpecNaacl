"""Execute actual Triton equations via CPU simulator; NOT GPU compilation."""
import math
import pytest
import torch
from test_reflex_kernel_equations import Pointer, kernel_source, pow2
from helper.fast_lk_reflex import lk_alpha_and_logit_gradient_without_loss
from helper.tree_verification import PackedTree, trace_verified_path


@pytest.mark.parametrize("contexts", [1, 4, 7, 8])
@pytest.mark.parametrize("dim", [4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("ties", [False, True])
def test_multicontext_fused_topk_actual_source(contexts, dim, dtype, ties):
    torch.manual_seed(82)
    tl, kernels = kernel_source()
    batch, vocab, k, bv = 2, 521, 4, 256
    tiles = math.ceil(vocab / bv)
    raw = torch.randn(batch, contexts, vocab * 2).to(dtype)[..., ::2]
    psi, state = torch.randn(batch, contexts, dim), torch.randn(batch, vocab, dim) * .2
    if ties:
        raw.zero_()
        state.zero_()
    maxima, sums = torch.empty(batch, contexts, tiles), torch.empty(batch, contexts, tiles)
    values, ids = torch.empty(batch, contexts, tiles, k), torch.empty(batch, contexts, tiles, k, dtype=torch.long)
    for b in range(batch):
        for tile in range(tiles):
            tl.ids = (tile, b)
            kernels._proposal_tiles(*map(Pointer, (raw, psi, state, maxima, sums, values, ids)),
                *raw.stride(), vocab, dim, contexts, k, tiles, bv, pow2(dim))
    probs, selected, norm = torch.empty(batch, contexts, k), torch.empty(batch, contexts, k, dtype=torch.long), torch.empty(batch, contexts, 2)
    for row in range(batch * contexts):
        tl.ids = (row,)
        kernels._proposal_merge(*map(Pointer, (maxima, sums, values, ids, probs, selected, norm)),
            contexts, vocab, k, tiles, pow2(tiles), pow2(tiles * k))
    z = torch.baddbmm(raw.float(), psi, state.transpose(1, 2))
    expected = z.softmax(-1).topk(k)
    if ties:
        assert torch.equal(selected, torch.arange(k).expand_as(selected))
    else:
        assert torch.equal(selected, expected.indices)
    torch.testing.assert_close(probs, expected.values, rtol=3e-6, atol=5e-8)
    reconstructed = (z - norm[..., :1]).exp() / norm[..., 1:]
    torch.testing.assert_close(reconstructed, z.softmax(-1), rtol=3e-6, atol=5e-8)


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("dim", [4, 8])
@pytest.mark.parametrize("extra_head", [False, True])
def test_fused_visited_update_actual_source(greedy, dim, extra_head):
    torch.manual_seed(5)
    tl, kernels = kernel_source()
    batch, vocab, cache, width, tiles, bv = 2, 521, 4, 3, 3, 256
    raw, psi, state = torch.randn(batch, cache, vocab).bfloat16(), torch.randn(batch, cache, dim), torch.randn(batch, vocab, dim) * .1
    z = torch.baddbmm(raw.float(), psi, state.transpose(1, 2))
    maximum = z.amax(-1)
    norm = torch.stack((maximum, (z - maximum[..., None]).exp().sum(-1)), -1)
    full = torch.randn(batch, 7, vocab + 13).softmax(-1)
    target = full.argmax(-1) if greedy else full
    mapping = torch.arange(vocab) + 3
    indices = torch.tensor([[0, 4, -1], [0, 1, 5]])
    contexts = torch.tensor([[0, 2, -1], [0, 1, -1]])
    if extra_head:
        contexts[1, 2] = 3
    valid = contexts >= 0
    expected = state.clone() * .99
    for b in range(batch):
        for j in range(width):
            if valid[b, j]:
                row, ctx = int(indices[b, j]), int(contexts[b, j])
                p = target[b, row].eq(mapping).float() if greedy else target[b, row, mapping]
                p /= p.sum() + 1e-8
                _, grad = lk_alpha_and_logit_gradient_without_loss(z[b, ctx].softmax(-1), p)
                expected[b] -= .05 * grad[:, None] * psi[b, ctx] / valid[b].sum()
    mass, stats, alpha = torch.empty(batch, width, tiles), torch.empty(batch, width, tiles, 2), torch.empty(batch, width)
    common = (vocab, dim, cache, width, *raw.stride(), target.stride(0), target.stride(1),
              0 if greedy else target.stride(2), *indices.stride(), *contexts.stride(), mapping.stride(0), greedy, 1e-8)
    tensors = tuple(map(Pointer, (raw, psi, norm, state, target, mapping, indices, contexts, mass, stats)))
    for mass_only in (True, False):
        for b in range(batch):
            for tile in range(tiles):
                tl.ids = (tile, b)
                kernels._path_stats(*tensors, *common, tiles, pow2(tiles), bv, pow2(dim), mass_only)
    for b in range(batch):
        for tile in range(tiles):
            tl.ids = (tile, b)
            kernels._path_update(*tensors, Pointer(alpha), *common, .05, .99, tiles, pow2(tiles), bv, pow2(dim))
    torch.testing.assert_close(state, expected, rtol=1e-5, atol=5e-7)
    assert torch.equal(alpha[~valid], torch.zeros_like(alpha[~valid]))


def test_fused_trace_source_eos_duplicates_and_unexpanded_leaf():
    tl, kernels = kernel_source()
    tree = PackedTree(torch.tensor([[-1, 0, 0, 1, 1, 3], [-1, 0, 0, 1, 1, 3]]),
        torch.tensor([[-1, 3, 3, 4, 5, 6], [-1, 3, 3, 4, 5, 6]]),
        torch.tensor([[0, 1, 2, 3, -1, -1], [0, 1, 2, 3, -1, -1]]), 3)
    samples = torch.tensor([[3, 4, 9, 6, 8, 2], [3, 2, 4, 6, 8, 7]])
    expected = trace_verified_path(tree, samples, 2)
    outputs = [torch.empty_like(x) for x in (expected.tokens, expected.packed_indices, expected.feedback_contexts, expected.lengths)]
    for b in range(2):
        tl.ids = (b,)
        kernels._trace_path(*map(Pointer, (tree.parents, tree.tokens, tree.feedback_contexts, samples, *outputs)),
                           6, 4, 2, *outputs[0].stride(), 8)
    for actual, reference in zip(outputs, (expected.tokens, expected.packed_indices, expected.feedback_contexts, expected.lengths)):
        assert torch.equal(actual, reference)
    mask = torch.empty(2, 1, 6, 17)
    for b in range(2):
        for row in range(6):
            tl.ids = (row, b)
            kernels._tree_mask(Pointer(tree.parents), Pointer(mask), 6, 11, 4, torch.finfo(mask.dtype).min, 32)
    torch.testing.assert_close(mask, tree.attention_mask(11, mask.dtype), rtol=0, atol=0)
