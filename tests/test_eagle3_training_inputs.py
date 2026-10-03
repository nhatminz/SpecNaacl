"""Inference rollout -> real EAGLE draft backward, using tiny CPU weights."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from helper.eagle3_specforge import rollout_tensor_for_training


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "third_party/SpecForge/specforge"


@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.bfloat16])
def test_inference_conversion_preserves_values_and_supports_backward(dtype):
    with torch.inference_mode():
        original = torch.arange(12).reshape(3, 4).to(dtype).clone()
        # The helper must disable inference mode for the clone, not just clone
        # under an inherited inference context.
        converted = rollout_tensor_for_training(original)
    assert torch.is_inference(original)
    assert not torch.is_inference(converted)
    assert not converted.requires_grad
    assert converted.dtype == original.dtype
    assert converted.device == original.device
    assert converted.data_ptr() != original.data_ptr()
    torch.testing.assert_close(converted, original, rtol=0, atol=0)
    if dtype == torch.long:
        layer = torch.nn.Embedding(12, 2)
    else:
        layer = torch.nn.Linear(4, 2, bias=False).to(dtype)
    layer(converted).float().sum().backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_normal_input_is_reused_without_detaching_graph():
    leaf = torch.randn(3, requires_grad=True)
    original = leaf * 2
    assert rollout_tensor_for_training(original) is original
    original.sum().backward()
    torch.testing.assert_close(leaf.grad, torch.full_like(leaf, 2))


@pytest.fixture
def cpu_training_model(monkeypatch):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from specforge.modeling.draft import llama3_eagle as eagle

    # Windows CPU has neither Triton nor a C++ compiler. Unwrap only the tiny
    # fixture's compiled RoPE/norm helpers, without changing production code.
    for cls in (eagle.LlamaRMSNorm, eagle.LlamaRotaryEmbedding):
        monkeypatch.setattr(cls, "forward", cls.forward._torchdynamo_orig_callable)
    monkeypatch.setattr(eagle, "apply_rotary_pos_emb",
                        eagle.apply_rotary_pos_emb._torchdynamo_orig_callable)

    # Use SpecForge's own reference loss rather than its CUDA-only Triton kernel.
    # Execute the real OnlineEagle3Model source with only that import replaced;
    # feature projection, SDPA, teacher, TTT and LK equations stay real.
    loss_tree = ast.parse((SPEC / "core/loss.py").read_text(encoding="utf-8"))
    reference = next(node for node in loss_tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss")
    reference.decorator_list = []
    scope = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=[reference], type_ignores=[]), "loss.py", "exec"), scope)
    scope["LogSoftmaxLoss"] = SimpleNamespace(apply=scope["_compute_loss"])
    model_tree = ast.parse((SPEC / "algorithms/eagle3/model.py").read_text(encoding="utf-8"))
    model_tree.body = [node for node in model_tree.body
                       if not (isinstance(node, ast.ImportFrom)
                               and node.module == "specforge.core.loss")]
    exec(compile(model_tree, "eagle3/model.py", "exec"), scope)

    torch.manual_seed(42)
    cfg = LlamaConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=4, num_hidden_layers=1,
                      max_position_embeddings=2048, vocab_size=32, draft_vocab_size=16,
                      target_hidden_size=16, pretraining_tp=1, tie_word_embeddings=False)
    draft = eagle.LlamaForCausalLMEagle3(cfg, attention_backend="sdpa")
    draft.t2d[16:] = False
    target = torch.nn.Module()
    target.lm_head = torch.nn.Linear(16, 32, bias=False)
    # Concentrated full-vocab teacher: argmax=0 is inside the selected draft
    # vocab, with probabilities straddling the draft's near-uniform outputs.
    # This exercises nonzero gradients for acceptance-only LK objectives too.
    with torch.no_grad():
        target.lm_head.weight.zero_()
        target.lm_head.weight[0, 0] = 5
    target.requires_grad_(False)
    online = scope["OnlineEagle3Model"](draft, length=7, attention_backend="sdpa")
    return SimpleNamespace(draft_model=draft, target_model=target,
                           specforge_training_model=online, device=torch.device("cpu"))


def training_entrypoint():
    # Importing grpo_speculative runs CUDA/model/reward initialization. Execute
    # just the unchanged production function instead, as other source tests do.
    tree = ast.parse((ROOT / "grpo_speculative.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "training_eagle3_specforge")
    scope = {"torch": torch, "repeated_generate_nums": 2,
             "_get_base_causal_lm": lambda model: model,
             "rollout_tensor_for_training": rollout_tensor_for_training}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "grpo_speculative.py", "exec"), scope)
    return scope["training_eagle3_specforge"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lk_loss_type", [None, "alpha", "tv", "lambda"])
@pytest.mark.parametrize("token_budget", [None, 5])
def test_rollout_training_matches_normal_inputs_loss_and_gradients(
    cpu_training_model, dtype, lk_loss_type, token_budget,
):
    model = cpu_training_model
    model.draft_model.to(dtype)
    model.target_model.to(dtype)
    model.dtype = dtype
    model.specforge_training_model.lk_loss_type = lk_loss_type
    outputs = {
        "all_draft_input_states": [torch.randn(16, 48, dtype=dtype) for _ in range(2)],
        "all_target_hidden_states": [torch.randn(16, 16, dtype=dtype) for _ in range(2)],
        "all_draft_input_ids": [torch.randint(0, 32, (16,)) for _ in range(2)],
    }
    for row in outputs["all_target_hidden_states"]:
        row[:, 0] = 1
    with torch.inference_mode():
        inference_outputs = {key: [row.clone() for row in rows] for key, rows in outputs.items()}
    assert all(torch.is_inference(row) for rows in inference_outputs.values() for row in rows)
    training = training_entrypoint()
    prompt_mask = torch.ones(1, 4, dtype=torch.long)
    expected = training(model, outputs, prompt_mask, token_budget)
    grads = {name: param.grad.clone() for name, param in model.draft_model.named_parameters()
             if param.grad is not None}
    assert "fc.weight" in grads
    assert torch.count_nonzero(grads["fc.weight"]) > 0
    model.draft_model.zero_grad(set_to_none=True)
    # Observe actual boundary tensors as well as checking successful backward.
    def check_inputs(module, args, kwargs):
        for key in ("input_ids", "hidden_states", "target_hidden_for_compact"):
            assert not torch.is_inference(kwargs[key])
            assert not kwargs[key].requires_grad
        assert kwargs["target_head_weight"] is model.target_model.lm_head.weight
    hook = model.specforge_training_model.register_forward_pre_hook(check_inputs, with_kwargs=True)
    try:
        actual = training(model, inference_outputs, prompt_mask, token_budget)
    finally:
        hook.remove()
    assert actual == expected
    assert actual[-1] == (22 if token_budget is None else 5)
    for name, param in model.draft_model.named_parameters():
        if name in grads:
            assert torch.isfinite(param.grad).all()
            torch.testing.assert_close(param.grad, grads[name], rtol=0, atol=0)
        else:
            assert param.grad is None
    assert model.target_model.lm_head.weight.grad is None
    for key in outputs:
        for inference_row, normal_row in zip(inference_outputs[key], outputs[key]):
            assert torch.is_inference(inference_row)
            torch.testing.assert_close(inference_row, normal_row, rtol=0, atol=0)


def test_zero_budget_skips_training(cpu_training_model):
    model = cpu_training_model
    model.dtype = torch.float32
    with torch.inference_mode():
        outputs = {
            "all_draft_input_states": [torch.randn(16, 48)],
            "all_target_hidden_states": [torch.randn(16, 16)],
            "all_draft_input_ids": [torch.randint(0, 32, (16,))],
        }
    assert training_entrypoint()(model, outputs, torch.ones(1, 4), 0) == (0, 0, 0, 0, 0)
    assert all(param.grad is None for param in model.draft_model.parameters())
