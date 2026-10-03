#!/usr/bin/env python3
"""Wall-clock comparison of old/new accepted-path and finished-row handling.

Includes the required GPU/CPU synchronization for host metadata. Synthetic
only: cache sizes and path shapes are configurable; run on B200 before making
an end-to-end speed claim.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from helper.tree_verification import VerifiedPath
from helper import fast_lk_reflex_kernels as kernels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--past", type=int, default=512)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--finished", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if min(args.batch, args.width, args.past, args.layers, args.heads, args.head_dim,
           args.iterations) <= 0 or not 0 <= args.finished < args.batch:
        parser.error("invalid positive shape or finished count")
    return args


def legacy_pad(path, past, eos):
    lengths, chosen, tokens = path.host_bookkeeping(past)
    width = max(lengths)
    pad_mask = []
    for row in range(len(lengths)):
        budget, cursor = width - lengths[row], past
        out_idx, out_tok, out_mask = [], [], []
        for index, token in zip(chosen[row], tokens[row]):
            if budget > 0:
                while cursor != index and budget:
                    out_idx.append(cursor)
                    out_tok.append(eos)
                    out_mask.append(True)
                    cursor += 1
                    budget -= 1
                out_idx.append(index)
                out_tok.append(token)
                out_mask.append(False)
                cursor += 1
            else:
                out_idx.append(index)
                out_tok.append(token)
                out_mask.append(False)
        while budget:
            out_idx.append(cursor)
            out_tok.append(eos)
            out_mask.append(True)
            cursor += 1
            budget -= 1
        chosen[row], tokens[row] = out_idx, out_tok
        pad_mask.append(out_mask)
    return torch.tensor(tokens, device=path.tokens.device), torch.tensor(chosen, device=path.tokens.device), pad_mask


def device_pad(path, past, eos, workspace=None):
    lengths = path.lengths.cpu().tolist()
    tokens, indices, mask, last = path.padded_gpu(past, max(lengths), eos,
                                                 kernels=kernels, workspace=workspace)
    packet = torch.cat((tokens.gather(1, last).eq(eos).long(),
                        torch.where(mask, indices, -1)), dim=1).cpu().tolist()
    return tokens, indices, mask, packet


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(12)
    b, width, past = args.batch, args.width, args.past
    indices = torch.arange(width, device=device).mul(2).expand(b, -1).clone()
    tokens = torch.arange(10, 10 + width, device=device).expand(b, -1).clone()
    lengths = torch.tensor([width - row % min(width, 3) for row in range(b)], device=device)
    valid = torch.arange(width, device=device)[None, :] < lengths[:, None]
    path = VerifiedPath(torch.where(valid, tokens, -1), torch.where(valid, indices, -1),
                        torch.where(valid, indices, -1), lengths)
    workspace = [torch.empty((b, width), device=device,
                             dtype=torch.bool if index == 2 else torch.long) for index in range(3)]
    workspace.append(torch.empty((b, 1), device=device, dtype=torch.long))
    old_tokens, old_indices, old_mask = legacy_pad(path, past, 2)
    new_tokens, new_indices, new_mask, _ = device_pad(path, past, 2, workspace)
    assert torch.equal(old_tokens, new_tokens) and torch.equal(old_indices, new_indices)
    assert torch.equal(torch.tensor(old_mask, device=device), new_mask)

    def measure(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            started = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000)
        return statistics.median(samples)

    path_old_ms = measure(lambda: legacy_pad(path, past, 2))
    path_new_ms = measure(lambda: device_pad(path, past, 2, workspace))
    # A moderate cache tensor models layer x batch x heads x tokens x dim.
    # Snapshot outside timing so both paths read the same shape and data.
    cache = torch.randn(args.layers, b, args.heads, past + width, args.head_dim,
                        device=device, dtype=torch.bfloat16)
    finished = list(range(args.finished))
    keep = torch.arange(args.finished, b, device=device)

    def old_compact():
        layers = [cache[layer] for layer in range(args.layers)]
        for deleted in reversed(finished):
            layers = [torch.cat((layer[:deleted], layer[deleted + 1:]), dim=0)
                      for layer in layers]
        return layers

    def new_compact():
        return [cache[layer].index_select(0, keep) for layer in range(args.layers)]

    old = old_compact()
    new = new_compact()
    assert all(torch.equal(x, y) for x, y in zip(old, new))
    compact_old_ms = measure(old_compact)
    compact_new_ms = measure(new_compact)
    layer_list = [cache[layer] for layer in range(args.layers)]
    gather_positions = torch.arange(width, device=device).expand(b, -1).contiguous()

    def stacked_suffix_gather():
        stacked = torch.stack(layer_list, dim=0)
        gather = gather_positions[None, :, None, :, None].expand(
            args.layers, b, args.heads, width, args.head_dim)
        suffix = stacked[..., past:, :].gather(-2, gather)
        return [torch.cat((stacked[layer, ..., :past, :], suffix[layer]), dim=-2)
                for layer in range(args.layers)]

    def per_layer_suffix_gather():
        gather = gather_positions[:, None, :, None].expand(b, args.heads, width, args.head_dim)
        return [torch.cat((layer[..., :past, :], layer[..., past:, :].gather(-2, gather)), dim=-2)
                for layer in layer_list]

    assert all(torch.equal(x, y) for x, y in zip(stacked_suffix_gather(), per_layer_suffix_gather()))
    suffix_stacked_ms = measure(stacked_suffix_gather)
    suffix_per_layer_ms = measure(per_layer_suffix_gather)
    report = dict(device=torch.cuda.get_device_name(0), shape=vars(args),
                  path_host_roundtrip_ms=path_old_ms, path_gpu_metadata_ms=path_new_ms,
                  finished_per_row_concat_ms=compact_old_ms,
                  finished_batched_index_select_ms=compact_new_ms,
                  kv_stacked_suffix_gather_ms=suffix_stacked_ms,
                  kv_per_layer_suffix_gather_ms=suffix_per_layer_ms,
                  parity=True, note="Synthetic wall-clock only; full model rollout is the final decision gate.")
    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main(parse_args())
