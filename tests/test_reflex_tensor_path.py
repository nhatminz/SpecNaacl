"""Differential verifier/feedback tests; no model downloads or GPU required."""
import pytest
import torch

from helper.fast_lk_reflex import FastLKReflex, lk_alpha_and_logit_gradient_without_loss
from helper.tree_verification import PackedTree, VerifiedPath, pack_tree, trace_verified_path


def reference_path(tree, samples, eos):
    result = []
    for b in range(samples.shape[0]):
        current, path = 0, []
        while True:
            token = int(samples[b, current])
            path.append((token, current, int(tree.feedback_contexts[b, current])))
            if token == eos:
                break
            children = [i for i in range(1, samples.shape[1])
                        if tree.parents[b, i] == current and tree.tokens[b, i] == token]
            if not children:
                break
            current = children[0]
        result.append(path)
    return result


@pytest.mark.parametrize("seed", range(8))
def test_tensor_verifier_matches_legacy_matching_and_ancestor_mask(seed):
    generator = torch.Generator().manual_seed(seed)
    batch, count = 4, 21
    parents = torch.tensor([[-1] + [(i - 1) // 3 for i in range(1, count)]]).expand(batch, -1).clone()
    tokens = torch.randint(0, 5, (batch, count), generator=generator)
    contexts = torch.arange(count).expand(batch, -1).clone()
    contexts[:, 8:] = -1  # unexpanded leaf: bonus emitted but no draft feedback
    tree = PackedTree(parents, tokens, contexts, 5)
    samples = torch.randint(0, 5, (batch, count), generator=generator)
    expected = reference_path(tree, samples, 4)
    actual = trace_verified_path(tree, samples, 4)
    lengths, chosen, emitted = actual.host_bookkeeping(11)
    for b, path in enumerate(expected):
        assert lengths[b] == len(path)
        assert chosen[b] == [11 + i for _, i, _ in path]
        assert emitted[b] == [t for t, _, _ in path]
        assert actual.feedback_contexts[b, :len(path)].tolist() == [c for _, _, c in path]
        assert (actual.packed_indices[b, len(path):] == -1).all()
    padding = torch.tensor([[0, 2], [3, 7]])
    mask = tree.attention_mask(11, torch.float32, padding)
    expected_mask = torch.zeros_like(mask)
    expected_mask[..., 12:] = torch.finfo(torch.float32).min
    for b in range(batch):
        for row in range(count):
            node = row
            while node >= 0:
                expected_mask[b, 0, row, 11 + node] = 0
                node = int(parents[b, node])
    expected_mask[padding[:, 0], 0, :, padding[:, 1]] = torch.finfo(torch.float32).min
    torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)


def test_packing_preserves_order_and_rejects_orphans():
    parents = torch.tensor([[-1, -1, 0, 0, 1, 1]])
    contexts = torch.tensor([[1, 2, -1, -1, -1, -1]])
    tokens = torch.tensor([[5, 6, 7, 8, 9, 10]])
    packed = pack_tree(parents, contexts, torch.tensor([[1, 4, 5]]), tokens, 2)
    assert packed.parents.tolist() == [[-1, 0, 1, 1]]
    assert packed.tokens.tolist() == [[-1, 6, 9, 10]]
    with pytest.raises(RuntimeError, match="parent-closed"):
        pack_tree(parents, contexts, torch.tensor([[4]]), tokens, 2)


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("extra_head", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_visited_mean_uses_only_entered_heads_and_decay_once(greedy, dtype, extra_head, amp):
    torch.manual_seed(72)
    engine = FastLKReflex(feature_dim=4, weight_decay=0.2, backend="torch", feedback_scope="visited_path")
    engine.start(2, 17, 7, "cpu", max_contexts=4, max_path_length=3)
    engine.state.normal_(std=0.1)
    raw, hidden = torch.randn(2, 4, 17).to(dtype), torch.randn(2, 4, 7).to(dtype)
    mapping = torch.arange(17) + 2
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
        engine.propose(raw[:, :1], hidden[:, :1], 4, mapping, root=True)
        engine.propose(raw[:, 1:], hidden[:, 1:], 4, mapping)
    q = engine.correct(raw, hidden)
    psi = engine._feature(hidden)
    teacher = torch.randn(2, 7, 21).softmax(-1)
    target = teacher.argmax(-1) if greedy else teacher
    indices = torch.tensor([[0, 4, -1], [0, 1, 5]])
    contexts = torch.tensor([[0, 2, -1], [0, 1, -1]])  # second bonus leaf has no head
    if extra_head:
        contexts[1, 2] = 3  # different eligible counts: mean must be per trajectory
    valid = contexts >= 0
    expected = engine.state.clone() * (1 - engine.learning_rate * engine.weight_decay)
    for b in range(2):
        delta = torch.zeros_like(expected[b])
        for j in range(3):
            if valid[b, j]:
                row, ctx = int(indices[b, j]), int(contexts[b, j])
                p = target[b, row].eq(mapping).float() if greedy else target[b, row, mapping]
                p = p / (p.sum() + engine.eps)
                _, gradient = lk_alpha_and_logit_gradient_without_loss(q[b, ctx], p)
                delta += gradient[:, None] * psi[b, ctx]
        expected[b] -= engine.learning_rate * delta / valid[b].sum()
    # Counterfactual target rows deliberately changed: must have NO effect.
    for b in range(2):
        eligible = {int(indices[b, j]) for j in range(3) if valid[b, j]}
        for row in set(range(7)) - eligible:
            target[b, row] = mapping[8] if greedy else torch.randn_like(target[b, row]).softmax(-1)
    path = VerifiedPath(indices, indices, contexts, valid.sum(1))
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
        engine.update_visited(target, mapping, path, greedy=greedy)
    torch.testing.assert_close(engine.state, expected, rtol=2e-5, atol=3e-7)
    assert engine.finish().updates == 2
    assert engine._root_q is None  # no full q across verification
    pointers = [engine._feedback_raw.data_ptr(), engine._feedback_psi.data_ptr()]
    engine.remove_finished([0])
    engine.propose(raw[1:, :1], hidden[1:, :1], 4, mapping, root=True)
    assert pointers == [engine._feedback_raw.data_ptr(), engine._feedback_psi.data_ptr()]


def test_torch_root_propose_matches_old_cache_and_update_bitwise():
    torch.manual_seed(12)
    old, new = [FastLKReflex(backend="torch", weight_decay=0.1) for _ in range(2)]
    for engine in (old, new):
        engine.start(3, 31, 9, "cpu")
    new.state.copy_(old.state.normal_())
    raw, hidden, mapping = torch.randn(3, 1, 31), torch.randn(3, 1, 9), torch.arange(31)
    expected = old.correct(raw, hidden, cache_root=True).topk(4)
    values, ids, target_ids = new.propose(raw, hidden, 4, mapping, root=True)
    torch.testing.assert_close(values, expected.values, rtol=0, atol=0)
    assert torch.equal(ids, expected.indices) and torch.equal(target_ids, ids)
    teacher = torch.randn(3, 37).softmax(-1)
    old.update_from_target_probs(teacher, mapping)
    new.update_from_target_probs(teacher, mapping)
    torch.testing.assert_close(new.state, old.state, rtol=0, atol=0)


def test_extraction_and_feedback_have_no_host_tensor_read(monkeypatch):
    engine = FastLKReflex(feature_dim=4, backend='torch', feedback_scope='visited_path')
    engine.start(1, 17, 7, 'cpu', max_contexts=2, max_path_length=3)
    raw, hidden, mapping = torch.randn(1, 2, 17), torch.randn(1, 2, 7), torch.arange(17)
    engine.propose(raw[:, :1], hidden[:, :1], 3, mapping, root=True)
    engine.propose(raw[:, 1:], hidden[:, 1:], 3, mapping)
    tree = PackedTree(torch.tensor([[-1, 0, 1]]), torch.tensor([[-1, 3, 5]]), torch.tensor([[0, 1, -1]]), 2)
    samples, teacher = torch.tensor([[3, 5, 2]]), torch.randn(1, 3, 17).softmax(-1)
    def forbidden(*unused, **kwargs):
        raise AssertionError('host tensor read during GPU path/feedback')
    for name in ('cpu', 'tolist', 'item'):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    path = trace_verified_path(tree, samples, 2, workspace=engine.path_workspace)
    engine.update_visited(teacher, mapping, path)


def test_debug_path_validation_rejects_bad_context_before_state_write():
    engine = FastLKReflex(backend='torch', feedback_scope='visited_path')
    engine.start(1, 17, 7, 'cpu', max_path_length=2)
    engine.propose(torch.randn(1, 1, 17), torch.randn(1, 1, 7), 3, torch.arange(17), root=True)
    indices, contexts = torch.tensor([[0, 1]]), torch.tensor([[0, 19]])
    path = VerifiedPath(indices, indices, contexts, torch.tensor([2]))
    original = engine.state.clone()
    with pytest.raises(RuntimeError, match='invalid feedback context'):
        engine.update_visited(torch.randn(1, 2, 17).softmax(-1), torch.arange(17), path, validate=True)
    torch.testing.assert_close(engine.state, original, rtol=0, atol=0)
