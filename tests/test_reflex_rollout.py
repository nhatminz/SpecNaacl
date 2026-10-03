"""Fixed-seed full production rollout on tiny deterministic CPU model doubles.

CUDA transfer streams/default mask device are emulated only in this fixture.
The sampler, tree builder, verifier, padding/KV pruning and Reflex are REAL.
This is a rollout control-flow test, not model quality or CUDA validation.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time
import math

import pytest
import torch

from helper.fast_lk_reflex import FastLKReflex, reflex_or_baseline_probabilities, topk_compact_candidates
from helper.sampling import build_sampling_probs, sample_from_probs, sample_target_from_logits
from helper.tree_verification import pack_tree, trace_verified_path


class Cache:
    def __init__(self):
        self.layers = []

    def get_seq_length(self):
        return self.layers[0].keys.shape[-2] if self.layers else 0

    def crop(self, length):
        for layer in self.layers:
            layer.keys = layer.keys[..., :length, :]
            layer.values = layer.values[..., :length, :]

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            layer.keys = layer.keys.repeat_interleave(repeats, 0)
            layer.values = layer.values.repeat_interleave(repeats, 0)


def load_rollout(source=None, device="cpu"):
    path = Path(__file__).resolve().parents[1] / "helper/specualtive_generate.py"
    tree = ast.parse(path.read_text(encoding="utf-8") if source is None else source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    # Only a test-device substitution. Never changes the production file.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_attention_mask":
            node.args.defaults[1] = ast.Constant(device)
    scope = dict(torch=torch, time=time, math=math, deepcopy=deepcopy, DynamicCache=Cache,
        ThreadPoolExecutor=ThreadPoolExecutor, FastLKReflex=FastLKReflex,
        reflex_or_baseline_probabilities=reflex_or_baseline_probabilities,
        topk_compact_candidates=topk_compact_candidates, sample_target_from_logits=sample_target_from_logits,
        build_sampling_probs=build_sampling_probs, sample_from_probs=sample_from_probs,
        pack_tree=pack_tree, trace_verified_path=trace_verified_path)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), "exec"), scope)
    return scope["speculative_generate"]


class TinyModel:
    is_eagle3_specforge = True
    device, dtype, compact_vocab_size = torch.device("cpu"), torch.bfloat16, 17

    def __init__(self):
        generator = torch.Generator().manual_seed(121)
        self.embedding = torch.randn(17, 8, generator=generator).bfloat16()
        self.target_head = torch.nn.Linear(8, 17, bias=False).bfloat16()
        self.draft_head = torch.nn.Linear(8, 17, bias=False).bfloat16()
        with torch.no_grad():
            self.target_head.weight.copy_(torch.randn(17, 8, generator=generator) * .3)
            self.draft_head.weight.copy_(torch.randn(17, 8, generator=generator) * .3)
        self.calls, self.masks = 0, []
        self.target_model = SimpleNamespace(device=self.device, dtype=self.dtype,
                                           model=self.target_forward, lm_head=self.target_head)

    def target_forward(self, input_ids, attention_mask, past_key_values, **kwargs):
        self.calls += 1
        self.masks.append(attention_mask.clone())
        hidden = self.embedding[input_ids]
        keys = hidden.unsqueeze(1)
        if past_key_values.layers:
            keys = torch.cat((past_key_values.layers[0].keys, keys), -2)
        visible = (attention_mask[:, 0] == 0).to(self.dtype)
        # Attention depends on ancestry/history, not just current token.
        hidden = hidden + visible.matmul(keys[:, 0]) / visible.sum(-1, keepdim=True).clamp_min(1)
        past_key_values.layers = [SimpleNamespace(keys=keys, values=keys.clone())]
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=past_key_values)

    def __call__(self, hidden_states, input_ids, past_key_values=None, **kwargs):
        hidden = (hidden_states + self.embedding[input_ids]) * .5
        keys = hidden.unsqueeze(1)
        if past_key_values is not None:
            keys = torch.cat((past_key_values[0][0], keys), -2)
        return dict(hidden_states=hidden, next_feature_states=hidden,
                    past_key_values=[(keys, keys.clone())])

    def compute_compact_logits(self, hidden):
        return self.draft_head(hidden)

    def compact_to_target_ids(self, device):
        return torch.arange(17, device=device)


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("repeat", [1, 2])
def test_fixed_seed_rollout_tensor_reflex_preserves_legacy_verifier(sample, repeat, monkeypatch):
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: object())
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    generate = load_rollout()
    ids = torch.tensor([[0, 0, 4, 5], [3, 7, 8, 9]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    records = []
    for mode, scope in (("off", "root"), ("active", "root"), ("active", "visited_path")):
        torch.manual_seed(411)
        model = TinyModel()
        with torch.inference_mode():
            outputs = generate(model, ids, mask, SimpleNamespace(eos_token_id=16),
                do_sample=sample, repeated_generate_nums=repeat, max_length=18,
                verification_capacity=48, max_verification_num=16, max_draft_k=3,
                max_draft_token_length=3, min_draft_token_length=2,
                reflex_mode=mode, reflex_backend="torch", reflex_lr=0,
                reflex_feedback_scope=scope, return_all_draft_input=True)
        records.append((outputs, model.calls, model.masks))
    baseline = records[0]
    for candidate in records[1:]:
        assert candidate[1] == baseline[1]  # NO extra target forward
        for a, b in zip(candidate[2], baseline[2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key in ("generated_token_ids", "response_verification_rounds", "response_accepted_length_sum",
                    "total_accepted_draft_tokens", "total_proposed_draft_tokens"):
            assert candidate[0][key] == baseline[0][key]
        for key in ("all_draft_input_states", "all_target_hidden_states", "all_draft_input_ids"):
            for a, b in zip(candidate[0][key], baseline[0][key]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["off", "active"])
def test_per_layer_kv_gather_matches_stacked_rollout(mode, monkeypatch):
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: object())
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    generate = load_rollout()
    ids = torch.tensor([[0, 0, 4, 5], [3, 7, 8, 9]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    outcomes = []
    for strategy in ("stacked", "per_layer"):
        torch.manual_seed(511)
        model = TinyModel()
        with torch.inference_mode():
            output = generate(model, ids, mask, SimpleNamespace(eos_token_id=16),
                do_sample=True, repeated_generate_nums=2, max_length=18,
                verification_capacity=48, max_verification_num=16, max_draft_k=3,
                max_draft_token_length=3, min_draft_token_length=2,
                reflex_mode=mode, reflex_backend="torch", reflex_lr=.05,
                kv_gather_strategy=strategy)
        outcomes.append((output, model.calls))
    assert outcomes[0][1] == outcomes[1][1]
    for key in ("generated_token_ids", "response_verification_rounds", "response_accepted_length_sum"):
        assert outcomes[0][0][key] == outcomes[1][0][key]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA required")
@pytest.mark.parametrize("scope", ["root", "visited_path"])
def test_cuda_streamed_reflex_rollout_matches_single_stream(scope):
    class CudaTinyModel(TinyModel):
        device = torch.device("cuda:0")

        def __init__(self):
            super().__init__()
            self.embedding = self.embedding.cuda()
            self.target_head.cuda()
            self.draft_head.cuda()

    generate = load_rollout(device="cuda:0")
    ids = torch.tensor([[0, 0, 4, 5], [3, 7, 8, 9]], device="cuda")
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], device="cuda")
    outcomes = []
    for streamed in (False, True):
        torch.manual_seed(111)
        model = CudaTinyModel()
        with torch.inference_mode():
            output = generate(model, ids, mask, SimpleNamespace(eos_token_id=16),
                do_sample=True, repeated_generate_nums=2, max_length=18,
                verification_capacity=48, max_verification_num=16, max_draft_k=3,
                max_draft_token_length=3, min_draft_token_length=2,
                reflex_mode="active", reflex_backend="triton", reflex_lr=.05,
                reflex_feedback_scope=scope, reflex_update_stream=streamed)
        torch.cuda.synchronize()
        outcomes.append((output, model.calls))
    assert outcomes[0][1] == outcomes[1][1]
    for key in ("generated_token_ids", "response_verification_rounds", "response_accepted_length_sum"):
        assert outcomes[0][0][key] == outcomes[1][0][key]
