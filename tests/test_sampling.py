import torch

from helper.sampling import (
    build_sampling_probs,
    sample_from_probs,
    sample_target_from_logits,
)


def test_sampling_distribution_applies_temperature_top_p_and_top_k_once():
    logits = torch.tensor([[[3.0, 2.0, 1.0, -1.0]]])
    actual = build_sampling_probs(logits, temperature=2.0, top_p=0.8, top_k=2)
    base = torch.softmax(logits.float() / 2.0, dim=-1)
    # top-p=0.8 retains the first two sorted tokens; top-k=2 retains the same
    # support, followed by a single normalization.
    expected = torch.zeros_like(base)
    expected[..., :2] = base[..., :2]
    expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(-1), torch.ones(1, 1))


def test_sample_from_probs_uses_the_given_distribution_without_logits():
    probs = torch.tensor([[[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]])
    samples = sample_from_probs(probs)
    assert samples.tolist() == [[1], [0]]


def test_reflex_reuses_sampling_probs_without_extra_target_forward():
    class ReflexRecorder:
        def __init__(self):
            self.calls = 0
            self.root_probs = None

        def update_from_target_probs(self, root_probs, compact_to_target):
            self.calls += 1
            self.root_probs = root_probs

        def update_from_target_tokens(self, root_tokens, compact_to_target):
            raise AssertionError("sampling path unexpectedly used greedy supervision")

    target_forward_calls = 0

    def target_forward_once():
        nonlocal target_forward_calls
        target_forward_calls += 1
        return torch.tensor([[[2.0, 1.0, 0.0, -1.0]]])

    reflex = ReflexRecorder()
    logits = target_forward_once()
    _, probs = sample_target_from_logits(
        logits,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=3,
        eos_token_id=0,
        reflex=reflex,
        compact_to_target=torch.tensor([0, 2]),
    )
    assert target_forward_calls == 1
    assert reflex.calls == 1
    assert reflex.root_probs.data_ptr() == probs[:, 0, :].data_ptr()


def test_greedy_reflex_does_not_build_full_vocabulary_probabilities():
    class ReflexRecorder:
        def __init__(self):
            self.tokens = None

        def update_from_target_tokens(self, root_tokens, compact_to_target):
            self.tokens = root_tokens

        def update_from_target_probs(self, root_probs, compact_to_target):
            raise AssertionError("greedy path allocated target probabilities")

    reflex = ReflexRecorder()
    tokens, probs = sample_target_from_logits(
        torch.tensor([[[0.0, 3.0, 1.0, 2.0]]]),
        do_sample=False,
        temperature=1.0,
        top_p=0.95,
        top_k=None,
        eos_token_id=0,
        reflex=reflex,
        compact_to_target=torch.tensor([1, 3]),
    )
    assert tokens.tolist() == [[1]]
    assert probs is None
    assert reflex.tokens.tolist() == [1]
