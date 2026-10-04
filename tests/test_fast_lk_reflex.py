import torch
import pytest
from unittest import mock
import helper.fast_lk_reflex as reflex_module

from helper.fast_lk_reflex import (
    FastLKReflex,
    lk_alpha_and_logit_gradient,
    reflex_or_baseline_probabilities,
    topk_compact_candidates,
)
from helper.eagle3_specforge import _TargetVocabHead
from helper.method_config import resolve_method


def test_analytic_lk_gradient_matches_autograd():
    torch.manual_seed(3)
    logits = torch.randn(4, 11, dtype=torch.float64, requires_grad=True)
    p = torch.softmax(torch.randn(4, 11, dtype=torch.float64) + 0.37, dim=-1)
    q = torch.softmax(logits, dim=-1)
    eps = 1e-8
    loss = -torch.log(torch.minimum(p, q).sum(-1) + eps).sum()
    expected = torch.autograd.grad(loss, logits)[0]
    _, _, actual = lk_alpha_and_logit_gradient(q.detach(), p, eps)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)


def test_fast_state_is_not_parameter_or_optimizer_state():
    reflex = FastLKReflex(feature_dim=3)
    reflex.start(2, 7, 5, "cpu")
    assert not isinstance(reflex.state, torch.nn.Parameter)
    assert not isinstance(reflex.projection, torch.nn.Parameter)
    assert reflex.state.requires_grad is False
    assert reflex.projection.requires_grad is False
    model = torch.nn.Linear(5, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert id(reflex.state) not in optimizer_ids
    assert id(reflex.projection) not in optimizer_ids


def test_compatibility_head_does_not_register_or_freeze_draft_model():
    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(5, 7)
            self.vocab_size = 7
            self.d2t = torch.zeros(7, dtype=torch.long)

        def compute_logits(self, hidden):
            return self.projection(hidden)

    draft = Draft()
    head = _TargetVocabHead(draft)
    assert list(head.parameters()) == []
    for parameter in head.parameters():
        parameter.requires_grad = False
    assert all(parameter.requires_grad for parameter in draft.parameters())


def test_off_and_zero_active_use_identical_compact_proposals():
    torch.manual_seed(5)
    logits = torch.randn(2, 1, 13)
    hidden = torch.randn(2, 1, 7)
    mapping = torch.tensor([11, 3, 8, 4, 9, 2, 7, 5, 6, 0, 12, 1, 10])
    fastgrpo_method, fastgrpo_reflex = resolve_method("fastgrpo")
    specnaacl_method, specnaacl_reflex = resolve_method("specnaacl")
    assert (fastgrpo_method, fastgrpo_reflex) == ("fastgrpo", "off")
    assert (specnaacl_method, specnaacl_reflex) == ("specnaacl", "active")
    off = reflex_or_baseline_probabilities(logits, reflex=None)
    reflex = FastLKReflex(feature_dim=4, seed=9)
    reflex.start(2, 13, 7, "cpu")
    active = reflex_or_baseline_probabilities(
        logits, hidden, reflex, cache_root=True
    )
    torch.testing.assert_close(active, off, rtol=0, atol=0)
    off_values, off_compact, off_target = topk_compact_candidates(off, mapping, 5)
    active_values, active_compact, active_target = topk_compact_candidates(
        active, mapping, 5
    )
    torch.testing.assert_close(active_values, off_values, rtol=0, atol=0)
    assert torch.equal(active_compact, off_compact)
    assert torch.equal(active_target, off_target)


def test_reflex_mode_off_is_exact_baseline_behavior():
    logits = torch.tensor([[[1.0, -2.0, 0.5]]])
    expected = logits.float().softmax(-1)
    actual = reflex_or_baseline_probabilities(logits, reflex=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for opt-in GPU profile")
def test_opt_in_reflex_profile_reports_components():
    reflex = FastLKReflex(feature_dim=3, backend="triton", profile=True)
    reflex.start(2, 16, 8, "cuda")
    logits = torch.randn(2, 1, 16, device="cuda")
    hidden = torch.randn(2, 1, 8, device="cuda")
    mapping = torch.arange(16, device="cuda")
    reflex.propose(logits, hidden, 4, mapping, root=True)
    reflex.update_from_target_tokens(torch.tensor([1, 2], device="cuda"), mapping)
    sections = reflex.finish().profile_sections_ms
    assert sections["feature_projection_ms"] >= 0
    assert sections["proposal_ms"] >= 0
    assert sections["feedback_update_ms"] >= 0


def test_fused_nonzero_correction_matches_reference_einsum():
    torch.manual_seed(17)
    reflex = FastLKReflex(feature_dim=3, seed=4)
    reflex.start(2, 7, 5, "cpu")
    reflex.state.copy_(torch.randn_like(reflex.state) * 0.1)
    logits = torch.randn(2, 3, 7)
    hidden = torch.randn(2, 3, 5)
    psi = torch.nn.functional.normalize(
        hidden.float().matmul(reflex.projection), dim=-1, eps=1e-6
    )
    expected = (
        logits.float() + torch.einsum("bvd,bsd->bsv", reflex.state, psi)
    ).softmax(-1)
    actual = reflex.correct(logits, hidden)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)


def test_remove_keeps_state_aligned_and_no_leak():
    reflex = FastLKReflex(feature_dim=2)
    reflex.start(3, 4, 5, "cpu")
    reflex.state[0].fill_(10)
    reflex.state[1].fill_(20)
    reflex.state[2].fill_(30)
    reflex.remove_finished([1])
    assert reflex.active_trajectories == 2
    assert torch.all(reflex.state[0] == 10)
    assert torch.all(reflex.state[1] == 30)
    reflex.clear()
    reflex.start(1, 4, 5, "cpu")
    assert torch.count_nonzero(reflex.state) == 0


def test_root_update_uses_compact_mapping_and_no_autograd():
    reflex = FastLKReflex(
        feature_dim=2, learning_rate=0.1, seed=2, diagnostics=True
    )
    reflex.start(2, 3, 4, "cpu")
    logits = torch.randn(2, 1, 3)
    hidden = torch.randn(2, 1, 4)
    reflex.correct(logits, hidden, cache_root=True)
    before = reflex.state.clone()
    target = torch.softmax(torch.randn(2, 7), dim=-1)
    alpha, loss = reflex.update_from_target_probs(target, torch.tensor([1, 3, 6]))
    assert alpha.shape == loss.shape == (2,)
    assert not torch.equal(before, reflex.state)
    assert reflex.state.grad_fn is None


def test_lk_gradient_is_nonzero_for_mismatched_distributions():
    q = torch.tensor([[0.70, 0.20, 0.10]])
    p = torch.tensor([[0.10, 0.20, 0.70]])
    _, _, gradient = lk_alpha_and_logit_gradient(q, p)
    assert torch.count_nonzero(gradient).item() > 0


def test_small_analytic_sgd_step_decreases_lk_loss():
    logits = torch.tensor([[1.2, -0.3, 0.1]])
    p = torch.tensor([[0.10, 0.25, 0.65]])
    q = logits.softmax(-1)
    _, before, gradient = lk_alpha_and_logit_gradient(q, p)
    after_q = (logits - 0.05 * gradient).softmax(-1)
    _, after, _ = lk_alpha_and_logit_gradient(after_q, p)
    assert after.item() < before.item()


def test_target_distribution_is_conditioned_on_compact_vocabulary():
    reflex = FastLKReflex(
        feature_dim=2, learning_rate=0.1, seed=2, diagnostics=True
    )
    reflex.start(1, 3, 4, "cpu")
    reflex.correct(torch.tensor([[[1.0, 0.0, -1.0]]]), torch.randn(1, 1, 4), cache_root=True)
    full = torch.tensor([[0.05, 0.10, 0.15, 0.20, 0.25, 0.10, 0.15]])
    mapping = torch.tensor([1, 3, 6])
    expected_p = full.index_select(-1, mapping)
    expected_p = expected_p / (expected_p.sum(-1, keepdim=True) + reflex.eps)
    expected_alpha, expected_loss, _ = lk_alpha_and_logit_gradient(
        reflex._root_q, expected_p, reflex.eps
    )
    alpha, loss = reflex.update_from_target_probs(full, mapping)
    torch.testing.assert_close(alpha, expected_alpha)
    torch.testing.assert_close(loss, expected_loss)


def test_diagnostics_off_does_not_compute_or_report_lk_loss(monkeypatch):
    def forbidden_loss(*args, **kwargs):
        raise AssertionError("diagnostic loss was evaluated")

    monkeypatch.setattr(reflex_module, "lk_diagnostic_loss", forbidden_loss)
    reflex = FastLKReflex(feature_dim=2, learning_rate=0.1, diagnostics=False)
    reflex.start(1, 3, 4, "cpu")
    reflex.correct(torch.randn(1, 1, 3), torch.randn(1, 1, 4), cache_root=True)
    alpha, loss = reflex.update_from_target_probs(
        torch.softmax(torch.randn(1, 7), -1), torch.tensor([1, 3, 6])
    )
    stats = reflex.finish()
    assert alpha.shape == (1,)
    assert loss is None
    assert stats.alpha_sum is None
    assert stats.loss_sum is None
    assert stats.updates == 1


def test_greedy_update_uses_compact_tokens_without_full_vocab_one_hot(monkeypatch):
    def forbidden_one_hot(*args, **kwargs):
        raise AssertionError("full-vocabulary one-hot was allocated")

    monkeypatch.setattr(torch.nn.functional, "one_hot", forbidden_one_hot)
    reflex = FastLKReflex(feature_dim=2, learning_rate=0.1)
    reflex.start(2, 3, 4, "cpu")
    reflex.correct(torch.randn(2, 1, 3), torch.randn(2, 1, 4), cache_root=True)
    alpha, loss = reflex.update_from_target_tokens(
        torch.tensor([3, 8]), torch.tensor([1, 3, 6])
    )
    assert alpha.shape == (2,)
    assert loss is None
    assert reflex.state.shape == (2, 3, 2)


def test_compact_greedy_update_matches_full_vocab_reference():
    torch.manual_seed(29)
    mapping = torch.tensor([1, 3, 6])
    tokens = torch.tensor([3, 8])
    logits = torch.randn(2, 1, 3)
    hidden = torch.randn(2, 1, 4)
    optimized = FastLKReflex(feature_dim=2, learning_rate=0.1, seed=7)
    reference = FastLKReflex(feature_dim=2, learning_rate=0.1, seed=7)
    optimized.start(2, 3, 4, "cpu")
    reference.start(2, 3, 4, "cpu")
    optimized.correct(logits, hidden, cache_root=True)
    reference.correct(logits, hidden, cache_root=True)
    full_probs = torch.zeros(2, 9)
    full_probs.scatter_(1, tokens.unsqueeze(-1), 1.0)
    expected_alpha, expected_loss = reference.update_from_target_probs(
        full_probs, mapping
    )
    actual_alpha, actual_loss = optimized.update_from_target_tokens(tokens, mapping)
    torch.testing.assert_close(actual_alpha, expected_alpha, rtol=0, atol=0)
    assert actual_loss is expected_loss is None
    torch.testing.assert_close(optimized.state, reference.state, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_model_autocast_does_not_cast_fp32_fast_state_or_update(dtype):
    torch.manual_seed(67)
    reference = FastLKReflex(feature_dim=3, weight_decay=0.02, backend="torch")
    actual = FastLKReflex(feature_dim=3, weight_decay=0.02, backend="torch")
    for engine in (reference, actual):
        engine.start(2, 11, 7, "cpu")
    logits = torch.randn(2, 1, 11).to(dtype)
    hidden = torch.randn(2, 1, 7).to(dtype)
    target = torch.softmax(torch.randn(2, 17), -1)
    mapping = torch.arange(11)
    expected_q = reference.correct(logits, hidden, cache_root=True)
    expected_alpha, _ = reference.update_from_target_probs(target, mapping)
    with torch.autocast("cpu", dtype=dtype):
        actual_q = actual.correct(logits, hidden, cache_root=True)
        assert actual._root_psi.dtype == actual._root_q.dtype == torch.float32
        actual_alpha, _ = actual.update_from_target_probs(target, mapping)
    assert actual_q.dtype == actual.state.dtype == torch.float32
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_alpha, expected_alpha, rtol=0, atol=0)
    torch.testing.assert_close(actual.state, reference.state, rtol=0, atol=0)


def test_zero_active_equals_off_inside_model_autocast_and_inference_mode():
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        reflex = FastLKReflex(feature_dim=3)
        reflex.start(3, 17, 7, "cpu")
        logits = torch.randn(3, 1, 17).bfloat16()
        hidden = torch.randn(3, 1, 7).bfloat16()
        off = reflex_or_baseline_probabilities(logits)
        active = reflex.correct(logits, hidden, cache_root=True)
        torch.testing.assert_close(active, off, rtol=0, atol=0)
        reflex.update_from_target_tokens(torch.tensor([1, 2, 3]), torch.arange(17))
        reflex.remove_finished([1])
        assert reflex.state.shape[0] == 2
        reflex.clear()
        assert reflex.state is reflex._state_workspace is None


def test_backend_resolution_does_not_import_triton_on_cpu():
    with mock.patch.object(reflex_module.importlib.util, "find_spec",
                           side_effect=AssertionError("CPU probed Triton")):
        assert reflex_module.resolve_reflex_backend("auto", "cpu", 8) == "torch"
    with pytest.raises(RuntimeError, match="requires CUDA"):
        FastLKReflex(backend="triton").start(1, 3, 4, "cpu")
    with pytest.raises(ValueError, match="backend"):
        FastLKReflex(backend="unknown")
    with mock.patch.object(reflex_module.importlib.util, "find_spec", return_value=object()):
        assert reflex_module.resolve_reflex_backend("auto", "cuda:0", 8) == "triton"
        assert reflex_module.resolve_reflex_backend("torch", "cuda:0", 8) == "torch"
        assert reflex_module.resolve_reflex_backend("auto", "cuda:0", 65) == "torch"
    with mock.patch.object(reflex_module.importlib.util, "find_spec", return_value=None):
        assert reflex_module.resolve_reflex_backend("auto", "cuda:0", 8) == "torch"
        with pytest.raises(RuntimeError, match="requires CUDA"):
            reflex_module.resolve_reflex_backend("triton", "cuda:0", 8)


def test_finished_compaction_reuses_two_buffers_and_never_leaks_state():
    reflex = FastLKReflex(feature_dim=3)
    reflex.start(5, 7, 4, "cpu")
    for index in range(5):
        reflex.state[index].fill_(index + 1)
    buffers = {reflex.state.data_ptr(), reflex._state_workspace.data_ptr()}
    for finished, expected_rows in (([1, 3, 1], [1, 3, 5]), ([0], [3, 5]), ([1], [3])):
        previous = reflex.state.clone()
        reflex.remove_finished(finished)
        assert reflex.state.data_ptr() in buffers
        assert reflex._state_workspace.data_ptr() in buffers
        assert reflex.state[:, 0, 0].tolist() == expected_rows
        assert previous.data_ptr() not in buffers
    reflex.remove_finished([0])
    assert reflex.active_trajectories == 0
    reflex.clear()
    assert reflex._state_workspace is None
    reflex.start(2, 7, 4, "cpu")
    assert torch.count_nonzero(reflex.state) == 0


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_twenty_rounds_match_explicit_equations_with_compaction(greedy, weight_decay):
    torch.manual_seed(103)
    reflex = FastLKReflex(feature_dim=3, weight_decay=weight_decay, diagnostics=True, backend="torch")
    reflex.start(4, 13, 7, "cpu")
    expected_state = reflex.state.clone()
    mapping = torch.tensor([1, 3, 6, 8, 11, 15, 16, 18, 20, 23, 25, 27, 29])
    updates = 0
    for round_id in range(20):
        batch = reflex.active_trajectories
        hidden = torch.randn(batch, 1, 7)
        logits = torch.randn(batch, 1, 13)
        psi = torch.nn.functional.normalize(hidden.matmul(reflex.projection), dim=-1, eps=1e-6)
        expected_q = torch.baddbmm(logits, psi, expected_state.transpose(1, 2)).softmax(-1)
        actual_q = reflex.correct(logits, hidden, cache_root=True)
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        full = torch.softmax(torch.randn(batch, 31), -1)
        tokens = full.argmax(-1)
        if greedy:
            p = tokens[:, None].eq(mapping[None, :]).float()
        else:
            p = full.index_select(-1, mapping)
        p = p / (p.sum(-1, keepdim=True) + reflex.eps)
        alpha, loss, grad = lk_alpha_and_logit_gradient(expected_q.squeeze(1), p, reflex.eps)
        expected_state.baddbmm_(grad.unsqueeze(-1), psi.squeeze(1).unsqueeze(1),
                               beta=1 - reflex.learning_rate * weight_decay,
                               alpha=-reflex.learning_rate)
        if greedy:
            actual_alpha, actual_loss = reflex.update_from_target_tokens(tokens, mapping)
        else:
            actual_alpha, actual_loss = reflex.update_from_target_probs(full, mapping)
        torch.testing.assert_close(actual_alpha, alpha, rtol=0, atol=0)
        torch.testing.assert_close(actual_loss, loss, rtol=0, atol=0)
        torch.testing.assert_close(reflex.state, expected_state, rtol=0, atol=0)
        assert reflex.state.grad_fn is None
        updates += batch
        if round_id in (4, 11):
            reflex.remove_finished([1])
            expected_state = expected_state[[i for i in range(batch) if i != 1]]
    assert reflex.finish().updates == updates


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for real Triton kernel parity")
@pytest.mark.parametrize("vocab,feature_dim", [(19, 3), (32000, 8), (151936, 8)])
@pytest.mark.parametrize("greedy", [False, True])
def test_real_triton_multiround_parity_strides_autocast_and_compaction(vocab, feature_dim, greedy):
    pytest.importorskip("triton")
    torch.manual_seed(211)
    reference = FastLKReflex(feature_dim=feature_dim, backend="torch", diagnostics=True, weight_decay=0.02)
    fused = FastLKReflex(feature_dim=feature_dim, backend="triton", diagnostics=True, weight_decay=0.02)
    full_vocab = vocab + 17
    mapping = torch.arange(0, vocab * 2, 2, device="cuda") // 2 + 7
    # Non-contiguous mapping, logits, features and root verification views.
    mapping_storage = torch.zeros(vocab * 2, device="cuda", dtype=torch.long)
    mapping_storage[::2] = mapping
    mapping = mapping_storage[::2]
    for engine in (reference, fused):
        engine.start(3, vocab, 31, "cuda")
    for round_id in range(6):
        batch = fused.active_trajectories
        logits = torch.randn(batch, 1, vocab * 2, device="cuda", dtype=torch.bfloat16)[..., ::2]
        hidden = torch.randn(batch, 1, 62, device="cuda", dtype=torch.bfloat16)[..., ::2]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q_ref = reference.correct(logits, hidden, cache_root=True)
            q_fused = fused.correct(logits, hidden, cache_root=True)
            torch.testing.assert_close(q_fused, q_ref, rtol=5e-5, atol=2e-7)
            if round_id == 0:
                torch.testing.assert_close(q_fused, logits.float().softmax(-1), rtol=0, atol=0)
            torch.testing.assert_close(fused._root_psi, reference._root_psi, rtol=5e-5, atol=2e-6)
            full = torch.randn(batch, 2, full_vocab, device="cuda").softmax(-1)
            if greedy:
                tokens = full.argmax(-1)[:, 0]
                a_ref, l_ref = reference.update_from_target_tokens(tokens, mapping)
                a_fused, l_fused = fused.update_from_target_tokens(tokens, mapping)
            else:
                a_ref, l_ref = reference.update_from_target_probs(full[:, 0], mapping)
                a_fused, l_fused = fused.update_from_target_probs(full[:, 0], mapping)
        torch.testing.assert_close(a_fused, a_ref, rtol=5e-5, atol=2e-7)
        torch.testing.assert_close(l_fused, l_ref, rtol=5e-5, atol=2e-6)
        torch.testing.assert_close(fused.state, reference.state, rtol=1e-4, atol=2e-6)
        # Check branch contexts as well as root-only caching.
        branch_logits = torch.randn(batch, 4, vocab, device="cuda", dtype=torch.bfloat16)
        branch_hidden = torch.randn(batch, 4, 31, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(fused.correct(branch_logits, branch_hidden),
                                   reference.correct(branch_logits, branch_hidden), rtol=1e-4, atol=2e-7)
        if round_id == 2:
            for engine in (reference, fused):
                engine.remove_finished([1])
    assert reference.finish().updates == fused.finish().updates
