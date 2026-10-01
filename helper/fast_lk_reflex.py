"""Trajectory-local, optimizer-free LK Reflex state for EAGLE-3 rollout.

The class in this file is deliberately a plain Python object rather than an
``nn.Module``.  Its tensors are inference-time state: they are never model
parameters, never enter an optimizer, and never survive a rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import torch
import torch.nn.functional as F


def lk_alpha_and_logit_gradient(
    q: torch.Tensor,
    p: torch.Tensor,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return LK alpha, loss and analytic ``d loss / d logits(q)``.

    ``q`` must be a softmax distribution and ``p`` the teacher probability on
    the same vocabulary.  All reductions are batched over the last dimension.
    At the non-differentiable equality point we consistently choose ``m=0``.
    """
    if q.shape != p.shape:
        raise ValueError(f"p/q shape mismatch: p={tuple(p.shape)}, q={tuple(q.shape)}")
    q32 = q.float()
    p32 = p.float()
    alpha = torch.minimum(p32, q32).sum(dim=-1)
    mask = (q32 < p32).to(q32.dtype)
    selected_mass = (mask * q32).sum(dim=-1, keepdim=True)
    gradient = q32 * (selected_mass - mask) / (alpha.unsqueeze(-1) + float(eps))
    loss = -(alpha + float(eps)).log()
    return alpha, loss, gradient


@dataclass(frozen=True)
class ReflexStats:
    alpha_sum: float
    loss_sum: float
    updates: int
    profile_time_ms: float = 0.0


class FastLKReflex:
    """Low-rank-in-context fast state ``z = z0 + A @ psi``.

    A is stored in FP32 as ``[active_trajectory, compact_vocab, feature_dim]``.
    R and A have ``requires_grad=False`` and this class exposes no parameters.
    """

    def __init__(
        self,
        feature_dim: int = 8,
        learning_rate: float = 0.05,
        weight_decay: float = 0.0,
        seed: int = 42,
        eps: float = 1.0e-8,
        profile: bool = False,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if learning_rate < 0.0 or weight_decay < 0.0:
            raise ValueError("learning_rate and weight_decay must be non-negative")
        self.feature_dim = int(feature_dim)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.eps = float(eps)
        self.profile = bool(profile)
        self.projection: Optional[torch.Tensor] = None
        self.state: Optional[torch.Tensor] = None
        self._root_q: Optional[torch.Tensor] = None
        self._root_psi: Optional[torch.Tensor] = None
        self._alpha_sum: Optional[torch.Tensor] = None
        self._loss_sum: Optional[torch.Tensor] = None
        self._updates = 0
        self._profile_time_s = 0.0

    def _profile_start(self):
        if not self.profile:
            return None
        if self.state is not None and self.state.is_cuda:
            torch.cuda.synchronize(self.state.device)
        return time.perf_counter()

    def _profile_end(self, started) -> None:
        if started is None:
            return
        if self.state is not None and self.state.is_cuda:
            torch.cuda.synchronize(self.state.device)
        self._profile_time_s += time.perf_counter() - started

    @property
    def active_trajectories(self) -> int:
        return 0 if self.state is None else int(self.state.shape[0])

    def start(
        self,
        num_trajectories: int,
        compact_vocab_size: int,
        hidden_size: int,
        device: torch.device | str,
    ) -> None:
        """Create one zero fast state per response, once per rollout."""
        if num_trajectories <= 0 or compact_vocab_size <= 0 or hidden_size <= 0:
            raise ValueError("trajectory, vocabulary, and hidden sizes must be positive")
        device = torch.device(device)
        # Initialization is outside the per-round hot path. A device-local
        # generator makes R reproducible without copying it from CPU.
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        projection = torch.randn(
            (hidden_size, self.feature_dim),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        projection.mul_(hidden_size ** -0.5)
        projection.requires_grad_(False)
        self.projection = projection
        self.state = torch.zeros(
            (num_trajectories, compact_vocab_size, self.feature_dim),
            device=device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self._root_q = None
        self._root_psi = None
        self._alpha_sum = torch.zeros((), device=device, dtype=torch.float32)
        self._loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        self._updates = 0

    def _feature(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.projection is None:
            raise RuntimeError("FastLKReflex.start() must be called before correction")
        return F.normalize(hidden_states.float().matmul(self.projection), dim=-1, eps=1.0e-6)

    @torch.no_grad()
    def correct(
        self,
        compact_logits: torch.Tensor,
        native_hidden: torch.Tensor,
        *,
        cache_root: bool = False,
    ) -> torch.Tensor:
        """Apply A@psi and return probabilities; optionally cache root q/psi."""
        started = self._profile_start()
        if self.state is None:
            raise RuntimeError("FastLKReflex.start() must be called before correction")
        if compact_logits.shape[0] != self.state.shape[0]:
            raise ValueError("Reflex state is not aligned with the active batch")
        psi = self._feature(native_hidden)
        correction = torch.einsum("bvd,bsd->bsv", self.state, psi)
        corrected = compact_logits.float() + correction
        probabilities = corrected.softmax(dim=-1)
        if cache_root:
            if corrected.shape[1] != 1:
                raise ValueError("root correction expects exactly one proposal context")
            self._root_q = probabilities.squeeze(1)
            self._root_psi = psi.squeeze(1)
        self._profile_end(started)
        return probabilities

    @torch.no_grad()
    def update_from_target_probs(
        self,
        target_root_probs: torch.Tensor,
        compact_to_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one root-only analytic LK update using verification logits."""
        started = self._profile_start()
        if self.state is None or self._root_q is None or self._root_psi is None:
            raise RuntimeError("root q/psi were not cached before the Reflex update")
        mapping = compact_to_target.to(device=target_root_probs.device, dtype=torch.long)
        # Conditional compact-vocabulary LK: condition the exact target sampling
        # distribution on tokens controllable by the EAGLE compact head.
        p = target_root_probs.float().index_select(-1, mapping)
        p = p / (p.sum(dim=-1, keepdim=True) + self.eps)
        alpha, loss, gradient = lk_alpha_and_logit_gradient(self._root_q, p, self.eps)
        decay = 1.0 - self.learning_rate * self.weight_decay
        # baddbmm_ applies decay and the batched rank-one update without
        # materializing a [batch, vocabulary, feature] outer-product temporary.
        self.state.baddbmm_(
            gradient.unsqueeze(-1),
            self._root_psi.unsqueeze(1),
            beta=decay,
            alpha=-self.learning_rate,
        )
        self._alpha_sum.add_(alpha.sum())
        self._loss_sum.add_(loss.sum())
        self._updates += int(alpha.shape[0])
        self._root_q = None
        self._root_psi = None
        self._profile_end(started)
        return alpha, loss

    @torch.no_grad()
    def remove_finished(self, finished_indices) -> None:
        """Compact all completed responses once while preserving batch order."""
        if self.state is None:
            raise RuntimeError("FastLKReflex has not been started")
        if not finished_indices:
            return
        finished_set = {int(index) for index in finished_indices}
        if any(index < 0 or index >= self.state.shape[0] for index in finished_set):
            raise IndexError(finished_indices)
        keep_indices = [
            index for index in range(self.state.shape[0]) if index not in finished_set
        ]
        keep = torch.tensor(keep_indices, device=self.state.device, dtype=torch.long)
        self.state = self.state.index_select(0, keep)
        # A root cache belongs to the just-verified batch and has already been
        # consumed. Clearing defensively prevents accidental cross-round reuse.
        self._root_q = None
        self._root_psi = None

    def finish(self) -> ReflexStats:
        """Materialize aggregate scalars once, at rollout completion."""
        if self._updates == 0 or self._alpha_sum is None or self._loss_sum is None:
            return ReflexStats(
                alpha_sum=0.0, loss_sum=0.0, updates=0,
                profile_time_ms=self._profile_time_s * 1000.0,
            )
        return ReflexStats(
            alpha_sum=float(self._alpha_sum.item()),
            loss_sum=float(self._loss_sum.item()),
            updates=int(self._updates),
            profile_time_ms=self._profile_time_s * 1000.0,
        )

    def clear(self) -> None:
        self.projection = None
        self.state = None
        self._root_q = None
        self._root_psi = None
        self._alpha_sum = None
        self._loss_sum = None
        self._updates = 0
        self._profile_time_s = 0.0


def reflex_or_baseline_probabilities(
    raw_logits: torch.Tensor,
    native_hidden: Optional[torch.Tensor] = None,
    reflex: Optional[FastLKReflex] = None,
    *,
    cache_root: bool = False,
) -> torch.Tensor:
    """Use the same FP32 compact-proposal path in OFF and ACTIVE modes."""
    if reflex is None:
        return raw_logits.float().softmax(dim=-1)
    if native_hidden is None:
        raise ValueError("native_hidden is required when Reflex is active")
    return reflex.correct(raw_logits, native_hidden, cache_root=cache_root)


def topk_compact_candidates(probabilities, compact_to_target, k):
    """Shared OFF/ACTIVE compact top-k and fixed d2t mapping path."""
    values, compact_ids = torch.topk(probabilities, k=int(k), dim=-1)
    mapping = compact_to_target.to(probabilities.device, torch.long)
    return values, compact_ids, mapping[compact_ids]
