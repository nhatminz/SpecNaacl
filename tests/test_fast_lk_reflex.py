import torch

from helper.fast_lk_reflex import (
    FastLKReflex,
    lk_alpha_and_logit_gradient,
    reflex_or_baseline_probabilities,
)
from helper.eagle3_specforge import _TargetVocabHead


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


def test_zero_reflex_state_preserves_baseline_probabilities():
    torch.manual_seed(5)
    logits = torch.randn(2, 1, 13)
    hidden = torch.randn(2, 1, 7)
    expected = logits.float().softmax(-1)
    reflex = FastLKReflex(feature_dim=4, seed=9)
    reflex.start(2, 13, 7, "cpu")
    actual = reflex.correct(logits, hidden, cache_root=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reflex_mode_off_is_exact_baseline_behavior():
    logits = torch.tensor([[[1.0, -2.0, 0.5]]])
    expected = logits.softmax(-1)
    actual = reflex_or_baseline_probabilities(logits, reflex=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_remove_keeps_state_aligned_and_no_leak():
    reflex = FastLKReflex(feature_dim=2)
    reflex.start(3, 4, 5, "cpu")
    reflex.state[0].fill_(10)
    reflex.state[1].fill_(20)
    reflex.state[2].fill_(30)
    reflex.remove(1)
    assert reflex.active_trajectories == 2
    assert torch.all(reflex.state[0] == 10)
    assert torch.all(reflex.state[1] == 30)
    reflex.clear()
    reflex.start(1, 4, 5, "cpu")
    assert torch.count_nonzero(reflex.state) == 0


def test_root_update_uses_compact_mapping_and_no_autograd():
    reflex = FastLKReflex(feature_dim=2, learning_rate=0.1, seed=2)
    reflex.start(2, 3, 4, "cpu")
    logits = torch.randn(2, 1, 3)
    hidden = torch.randn(2, 1, 4)
    reflex.correct(logits, hidden, cache_root=True)
    before = reflex.state.clone()
    target = torch.randn(2, 7)
    alpha, loss = reflex.update_from_target_logits(target, torch.tensor([1, 3, 6]))
    assert alpha.shape == loss.shape == (2,)
    assert not torch.equal(before, reflex.state)
    assert reflex.state.grad_fn is None
