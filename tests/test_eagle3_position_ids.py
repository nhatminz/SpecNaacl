"""Actual EAGLE adapter/RoPE regression with tiny CPU weights, no GPU rollout."""

import ast
from pathlib import Path

import pytest
import torch

from helper.eagle3_specforge import Eagle3FastGRPOAdapter


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.long])
def test_rollout_prefill_creates_long_ids_without_changing_padding_positions(mask_dtype):
    # Execute only the production position-ID construction block: importing the
    # whole rollout module would require unrelated reward/runtime dependencies.
    tree = ast.parse((ROOT / "helper/specualtive_generate.py").read_text(encoding="utf-8"))
    generate = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "speculative_generate")
    start = next(index for index, node in enumerate(generate.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "position_ids"
                 and isinstance(node.value, ast.ListComp))
    end = next(index for index in range(start, len(generate.body))
               if isinstance(generate.body[index], ast.Assign)
               and isinstance(generate.body[index].value, ast.Call)
               and ast.unparse(generate.body[index].value.func) == "torch.stack")
    code = compile(ast.Module(body=generate.body[start:end + 1], type_ignores=[]),
                   "specualtive_generate.py", "exec")
    scope = {
        "torch": torch, "input_ids": torch.ones(4, 4, dtype=torch.long),
        "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1],
                                        [0, 1, 1, 1], [0, 0, 0, 0]], dtype=mask_dtype),
    }
    exec(code, scope)
    assert scope["position_ids"].dtype == torch.long
    assert torch.equal(scope["position_ids"], torch.tensor(
        [[0, 0, 0, 1], [0, 1, 2, 3], [0, 0, 1, 2], [0, 0, 0, 0]], dtype=torch.long))
    assert scope["past_position_ids"] == [1, 3, 2, -1]


@pytest.fixture
def tiny_adapter(monkeypatch):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from specforge.modeling.draft import llama3_eagle as eagle

    # Only unwrap helper compilation for ordinary CPU parity tests; production
    # decorators remain unchanged. A separate test exercises Dynamo below.
    for cls in (eagle.LlamaRMSNorm, eagle.LlamaRotaryEmbedding):
        monkeypatch.setattr(cls, "forward", cls.forward._torchdynamo_orig_callable)
    monkeypatch.setattr(eagle, "apply_rotary_pos_emb",
                        eagle.apply_rotary_pos_emb._torchdynamo_orig_callable)
    cfg = LlamaConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=4, num_hidden_layers=1,
                      max_position_embeddings=2048, vocab_size=32, draft_vocab_size=16,
                      target_hidden_size=16, pretraining_tp=1, tie_word_embeddings=False)
    torch.manual_seed(42)
    draft = eagle.LlamaForCausalLMEagle3(cfg, attention_backend="sdpa")
    # Bypass only checkpoint/target loading; use real draft weights, adapter
    # forward, SDPA, feature projection, RoPE and the flat KV-cache code.
    adapter = Eagle3FastGRPOAdapter.__new__(Eagle3FastGRPOAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.draft_model = draft
    adapter.config = cfg
    adapter.dtype = torch.float32
    return adapter


def prefill_inputs(dtype):
    positions = torch.tensor([[0, 0, 0, 1], [0, 1, 2, 3]], dtype=torch.long)
    mask = torch.zeros(2, 1, 4, 4, dtype=dtype).masked_fill(
        torch.ones(4, 4, dtype=torch.bool).triu(1), torch.finfo(dtype).min)
    mask[0, :, :, :2] = torch.finfo(dtype).min
    return {
        "hidden_states": torch.randn(2, 4, 48, dtype=dtype),
        "input_ids": torch.tensor([[0, 0, 3, 4], [5, 6, 7, 8]], dtype=torch.long),
        "attention_mask": mask,
        "position_ids": positions,
    }


def assert_outputs_identical(actual, expected):
    for key in ("hidden_states", "next_feature_states"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert torch.isfinite(actual[key]).all()
    for actual_layer, expected_layer in zip(actual["past_key_values"], expected["past_key_values"]):
        for actual_tensor, expected_tensor in zip(actual_layer, expected_layer):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("position_dtype", [torch.float32, torch.float64, torch.bfloat16,
                                           torch.int32, torch.int64])
@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
def test_adapter_prefill_and_cached_decode_match_long_reference(tiny_adapter, position_dtype, model_dtype):
    adapter = tiny_adapter.to(model_dtype)
    adapter.dtype = model_dtype
    inputs = prefill_inputs(model_dtype)
    expected = adapter(**inputs)
    supplied = dict(inputs, position_ids=inputs["position_ids"].to(position_dtype))
    actual = adapter(**supplied)
    assert_outputs_identical(actual, expected)
    assert actual["past_key_values"][0][0].shape[-2] == 4

    # Subsequent draft steps use native predicted features and flattened KV.
    decode_mask = torch.zeros(2, 1, 1, 5, dtype=model_dtype)
    decode_mask[0, :, :, :2] = torch.finfo(model_dtype).min
    decode = {
        "hidden_states": actual["next_feature_states"][:, -1:, :],
        "input_ids": torch.tensor([[9], [10]], dtype=torch.long),
        "attention_mask": decode_mask, "past_key_values": actual["past_key_values"],
        "position_ids": torch.tensor([[2], [4]], dtype=torch.long),
    }
    reference = adapter(**decode)
    converted = adapter(**dict(decode, position_ids=decode["position_ids"].to(position_dtype)))
    assert_outputs_identical(converted, reference)
    assert converted["past_key_values"][0][0].shape[-2] == 5
    assert torch.equal(supplied["position_ids"], inputs["position_ids"].to(position_dtype))


def test_adapter_default_positions_preserve_cache_offset_and_backward(tiny_adapter):
    inputs = prefill_inputs(torch.float32)
    inputs.pop("position_ids")
    expected = tiny_adapter(**dict(inputs, position_ids=torch.arange(4).unsqueeze(0)))
    actual = tiny_adapter(**inputs)
    assert_outputs_identical(actual, expected)
    decode = {
        "hidden_states": actual["hidden_states"][:, -1:, :],
        "input_ids": torch.tensor([[9], [10]], dtype=torch.long),
        "attention_mask": torch.zeros(2, 1, 1, 5),
        "past_key_values": actual["past_key_values"],
    }
    expected = tiny_adapter(**dict(decode, position_ids=torch.tensor([[4]], dtype=torch.long)))
    actual = tiny_adapter(**decode)
    assert_outputs_identical(actual, expected)
    actual["hidden_states"].square().mean().backward()
    assert tiny_adapter.draft_model.fc.weight.grad is not None
    assert torch.isfinite(tiny_adapter.draft_model.fc.weight.grad).all()
    uncached = tiny_adapter(**dict(inputs, use_cache=False))
    assert uncached["past_key_values"] == []


def test_float_positions_work_with_dynamo_rotary_compilation(tiny_adapter, monkeypatch):
    from specforge.modeling.draft import llama3_eagle as eagle

    # Exercise real Dynamo tracing/fake tensor indexing with fixed test shapes.
    # Dynamic symbolic guards may invoke MSVC even with backend="eager" on
    # Windows; no production compile decorator/configuration is changed.
    monkeypatch.setattr(eagle, "apply_rotary_pos_emb",
                        torch.compile(eagle.apply_rotary_pos_emb, backend="eager", dynamic=False))
    inputs = prefill_inputs(torch.float32)
    reference = tiny_adapter(**inputs)
    actual = tiny_adapter(**dict(inputs, position_ids=inputs["position_ids"].float()))
    assert_outputs_identical(actual, reference)
