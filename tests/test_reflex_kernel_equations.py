"""CPU simulation of the *actual* tiled kernel source, not a CUDA/JIT test.

The small tl subset below executes scalar/vector memory semantics with Torch.
It catches indexing, padding, stride and LK equation errors without Triton.
GPU compilation/numerics/performance still require the real CUDA parity tests.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from helper.fast_lk_reflex import lk_alpha_and_logit_gradient_without_loss


class Pointer:
    def __init__(self, tensor, offset=None):
        self.storage = tensor.as_strided(
            (tensor.untyped_storage().nbytes() // tensor.element_size(),), (1,), storage_offset=0,
        )
        self.offset = torch.as_tensor(tensor.storage_offset() if offset is None else offset)

    def __add__(self, offset):
        result = object.__new__(Pointer)
        result.storage, result.offset = self.storage, self.offset + offset
        return result


class CpuTL:
    constexpr = int
    float32 = torch.float32
    int64 = torch.int64

    def __init__(self):
        self.ids = ()

    def program_id(self, axis):
        return torch.tensor(self.ids[axis], dtype=torch.int32)

    arange = staticmethod(torch.arange)
    sqrt = staticmethod(torch.sqrt)
    div_rn = staticmethod(torch.div)

    @staticmethod
    def sum(value, axis):
        return value.sum(dim=axis)

    @staticmethod
    def minimum(x, y):
        return torch.minimum(torch.as_tensor(x), torch.as_tensor(y))

    @staticmethod
    def maximum(x, y):
        return torch.maximum(torch.as_tensor(x), torch.as_tensor(y))

    @staticmethod
    def where(condition, x, y):
        return torch.where(condition, x, y)

    @staticmethod
    def load(ptr, mask=True, other=0):
        mask = torch.broadcast_to(torch.as_tensor(mask), ptr.offset.shape)
        result = torch.full(ptr.offset.shape, other, dtype=ptr.storage.dtype)
        result[mask] = ptr.storage[ptr.offset[mask]]
        return result

    @staticmethod
    def store(ptr, value, mask=True):
        mask = torch.broadcast_to(torch.as_tensor(mask), ptr.offset.shape)
        value = torch.broadcast_to(torch.as_tensor(value), ptr.offset.shape)
        ptr.storage[ptr.offset[mask]] = value[mask]


def kernel_source():
    source = Path(__file__).resolve().parents[1] / "helper/fast_lk_reflex_kernels.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name.startswith("_")]
    for node in functions:
        node.decorator_list = []
    tl = CpuTL()
    scope = {"tl": tl}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    return tl, SimpleNamespace(**{key: value for key, value in scope.items() if key.startswith("_")})


def pow2(value):
    return 1 << (value - 1).bit_length()


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("vocab", [19, 2053])
@pytest.mark.parametrize("learning_rate,weight_decay", [(0.05, 0.0), (0.1, 0.2), (0.0, 0.0)])
def test_tiled_source_matches_projection_correction_and_update(greedy, vocab, learning_rate, weight_decay):
    torch.manual_seed(307)
    tl, kernels = kernel_source()
    batch, contexts, dim, hidden_size = 2, 3, 3, 7
    hidden = torch.randn(batch, contexts, hidden_size * 2).bfloat16()[..., ::2]
    projection = torch.randn(hidden_size, dim)
    psi = torch.empty(batch, contexts, dim)
    for row in range(batch * contexts):
        tl.ids = (row,)
        kernels._feature_kernel(Pointer(hidden), Pointer(projection), Pointer(psi),
                                *hidden.stride(), contexts, hidden_size, dim, pow2(hidden_size), pow2(dim))
    reference_psi = torch.nn.functional.normalize(hidden.float().matmul(projection), dim=-1, eps=1e-6)
    torch.testing.assert_close(psi, reference_psi, rtol=2e-6, atol=2e-7)
    state = torch.randn(batch, vocab, dim) * 0.1
    logits = torch.randn(batch, contexts, vocab * 2).bfloat16()[..., ::2]
    corrected = torch.empty(batch, contexts, vocab)
    for b in range(batch):
        for context in range(contexts):
            for tile in range(math.ceil(vocab / 256)):
                tl.ids = (tile, context, b)
                kernels._correction_kernel(Pointer(logits), Pointer(psi), Pointer(state), Pointer(corrected),
                                           *logits.stride(), vocab, dim, contexts, 256, pow2(dim))
    reference_corrected = torch.baddbmm(logits.float(), psi, state.transpose(1, 2))
    torch.testing.assert_close(corrected, reference_corrected, rtol=2e-5, atol=5e-7)
    q = reference_corrected.softmax(-1)[:, 0]
    root_psi = psi[:, 0]
    full_vocab = vocab + 17
    full = torch.randn(batch, 2, full_vocab).softmax(-1)[:, 0]
    mapping = torch.arange(vocab * 2)[::2] // 2 + 7
    # One controllable and one out-of-compact token, with non-contiguous strides.
    tokens = torch.tensor([[8, 0], [full_vocab - 1, 0]])[:, 0]
    target = tokens if greedy else full
    p = tokens[:, None].eq(mapping).float() if greedy else full.index_select(-1, mapping)
    eps = 1e-8
    p = p / (p.sum(-1, keepdim=True) + eps)
    alpha_ref, gradient = lk_alpha_and_logit_gradient_without_loss(q, p, eps)
    expected_state = state.clone()
    decay = 1 - learning_rate * weight_decay
    expected_state.baddbmm_(gradient.unsqueeze(-1), root_psi.unsqueeze(1), beta=decay, alpha=-learning_rate)
    tiles = math.ceil(vocab / 2048)
    mass, stats, alpha = torch.empty(batch, tiles), torch.empty(batch, tiles, 2), torch.empty(batch)
    ts0, ts1 = target.stride(0), 0 if greedy else target.stride(1)
    for b in range(batch):
        for tile in range(tiles):
            tl.ids = (tile, b)
            kernels._teacher_mass_kernel(Pointer(target), Pointer(mapping), Pointer(mass), vocab,
                                         ts0, ts1, mapping.stride(0), greedy, tiles, 2048)
    for b in range(batch):
        for tile in range(tiles):
            tl.ids = (tile, b)
            kernels._lk_stats_kernel(Pointer(q), Pointer(target), Pointer(mapping), Pointer(mass), Pointer(stats),
                                    *q.stride(), vocab, ts0, ts1, mapping.stride(0), greedy, eps,
                                    tiles, pow2(tiles), 2048)
    for b in range(batch):
        for tile in range(math.ceil(vocab / 128)):
            tl.ids = (tile, b)
            kernels._state_update_kernel(Pointer(state), Pointer(q), Pointer(root_psi), Pointer(target),
                                        Pointer(mapping), Pointer(mass), Pointer(stats), Pointer(alpha),
                                        *q.stride(), *root_psi.stride(), vocab, dim, ts0, ts1,
                                        mapping.stride(0), greedy, eps, learning_rate, decay,
                                        tiles, pow2(tiles), 128, pow2(dim))
    torch.testing.assert_close(alpha, alpha_ref, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(state, expected_state, rtol=1e-5, atol=5e-7)
