"""Opt-in generation telemetry shared by the two speculative engines."""
from collections import defaultdict
import time

import torch


def response_limit(max_response_tokens):
    if max_response_tokens is not None and max_response_tokens < 1:
        raise ValueError("max_response_tokens must be positive")
    return max_response_tokens


def validate_response_context(model, padded_prompt_length, response_cap, draft_depth):
    limit = getattr(model.target_model.config, "max_position_embeddings", None)
    if response_cap and limit and padded_prompt_length + response_cap + draft_depth > limit:
        raise ValueError("prompt + response cap + speculative tree exceeds model context")


def cap_path(path, remaining):
    remaining = torch.as_tensor(remaining, device=path.lengths.device)
    torch.minimum(path.lengths, remaining, out=path.lengths)
    valid = torch.arange(path.tokens.shape[1], device=remaining.device) < path.lengths[:, None]
    for field in (path.tokens, path.packed_indices, path.feedback_contexts):
        field.masked_fill_(~valid, -1)
    return path


def storage_bytes(tensors):
    storages = {}
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            storage = tensor.untyped_storage()
            storages[tensor.device, storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def cache_bytes(cache):
    if hasattr(cache, "statistics"):
        return cache.statistics()["kv_cache_bytes"]
    layers = ((layer.keys, layer.values) for layer in cache.layers) if hasattr(cache, "layers") else cache
    return storage_bytes(tensor for layer in layers for tensor in layer)


PHASES = (
    "prefill", "draft_proposal", "draft_advance", "target_verification", "target_sampling",
    "scheduling_batching", "tree_attention_mask", "verified_path_tracing",
    "kv_cache_management", "spark_feature_projection", "spark_correction_application",
    "spark_corrected_proposal_inclusive", "spark_feedback_preparation",
    "spark_correction_update", "spark_projector_gradient", "spark_update_stream_wait",
    "other_generation_overhead",
)


class GenerationTelemetry:
    def __init__(self, mode, device, responses_per_prompt, round_frequency, *, spark=False):
        self.enabled = mode in ("clean", "detailed")
        self.detailed = mode == "detailed"
        self.device = torch.device(device)
        self.repeats = responses_per_prompt or 1
        self.spark = spark
        self.frequency = round_frequency
        self.events = []
        self.rows = []
        self.cache = {"target_kv_bytes": None, "draft_kv_bytes": None}
        self.started = time.perf_counter()
        self.active_groups_sum = 0
        if self.enabled and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def begin(self, name):
        if not self.detailed:
            return None
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
        else:
            event = time.perf_counter()
        return name, event

    def end(self, ticket):
        if ticket is not None:
            name, start = ticket
            self.events.append((name, start, self.begin(name)[1]))

    def observe_cache(self, kind, cache):
        if self.detailed:
            key = f"{kind}_kv_bytes"
            self.cache[key] = max(self.cache[key] or 0, cache_bytes(cache))

    def samples(self, index):
        return self.detailed and self.frequency > 0 and (index - 1) % self.frequency == 0

    def round(self, index, owners, prompt_lengths, accepted_so_far, accepted, finished,
              budget, depth, branches, feedback=None, physical_rows=None, max_budget=None):
        if not self.enabled:
            return
        active_groups = len({owner // self.repeats for owner, count in zip(owners, accepted) if count})
        self.active_groups_sum += active_groups
        if not self.samples(index):
            return
        actual_rows = len(owners) if physical_rows is None else physical_rows
        self.rows.append(dict(
            verification_batch=index, active_responses=sum(count > 0 for count in accepted), active_prompt_groups=active_groups,
            verification_budget_per_response=budget if max_budget is None else max_budget,
            proposed_draft_tokens=budget * actual_rows,
            verified_tree_nodes=(budget + 1) * actual_rows, draft_depth=depth,
            branching_factor=branches, accepted_tokens=sum(accepted),
            accepted_draft_tokens=sum(max(0, count - 1) for count in accepted),
            completed_responses=sum(count > 0 and done for count, done in zip(accepted, finished)), context_lengths=[
                prompt_lengths[owner // self.repeats] + 1 + accepted_so_far[owner] - accepted[i]
                for i, owner in enumerate(owners) if accepted[i]], feedback=feedback,
        ))

    def finish(self, result, *, opd_memory=None, prompt_lengths=None):
        if not self.enabled:
            return result
        # One boundary sync. It also includes rollout-local cleanup queued by
        # the correction stream's consumer. Event resolution is outside wall.
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        wall = time.perf_counter() - self.started
        phases = dict.fromkeys(f"{name}_time_ms" for name in PHASES)
        totals = defaultdict(float)
        for name, start, end in self.events:
            totals[f"{name}_time_ms"] += (start.elapsed_time(end) if self.device.type == "cuda"
                                          else (end - start) * 1000)
        phases.update(totals)
        for name in PHASES:
            if name.startswith("spark_") and not self.spark:
                phases[f"{name}_time_ms"] = 0.0
        if self.spark:
            old_names = {
                "opd_feature_ms": "spark_feature_projection",
                "proposal_ms": "spark_corrected_proposal_inclusive",
                "opd_update_ms": "spark_correction_update",
                "opd_projector_grad_ms": "spark_projector_gradient",
                "opd_wait_ms": "spark_update_stream_wait",
            }
            sections = result.get("opd_profile_sections_ms") or {}
            for name, field in old_names.items():
                if name in sections:
                    phases[f"{field}_time_ms"] = sections[name]
            prep = [value for name, value in sections.items()
                    if name in ("opd_state_select_ms", "opd_teacher_extract_ms", "opd_union_loss_ms")]
            if prep:
                phases["spark_feedback_preparation_time_ms"] = sum(prep)
            for key in ("kv_batch_compaction_ms", "kv_suffix_compaction_ms"):
                if key in sections:
                    field = "kv_cache_management_time_ms"
                    phases[field] = (phases[field] or 0) + sections[key]
        for row in self.rows:
            feedback = row.pop("feedback")
            states, updates = (feedback.cpu().tolist() if feedback is not None else
                               (None, None) if self.spark else (0, 0))
            row.update(spark_feedback_states=states, spark_updates=updates)
        memory = dict(self.cache)
        memory.update(opd_memory or dict(correction_matrix_bytes=0, correction_matrix_count=0,
            dense_fp32_correction_bytes=0, projector_bytes=0, correction_workspace_bytes=0))
        if self.device.type == "cuda":
            memory.update(generation_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                          generation_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        result.update(total_time_cost=wall, prompt_token_lengths=prompt_lengths,
                      active_prompt_group_rounds=self.active_groups_sum,
                      generation_profile=dict(phases, generation_wall_time_ms=wall * 1000),
                      generation_memory=memory, verification_rounds=self.rows)
        return result
