import torch

from helper.sampling import build_sampling_probs, sample_from_probs


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
