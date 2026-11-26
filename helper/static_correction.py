"""Synchronous, rollout-local corrections to SPARK's draft logits."""
import math
import os

import torch


STATIC_DEFAULTS = dict(opd_static_correction='none', opd_static_alpha=1.,
                       opd_static_count_mode='miss', opd_cache_order=1,
                       opd_cache_slots=1, opd_cache_bonus=1.)
STATIC_MODES = ('none', 'token_freq', 'context_cache')
CACHE_BUCKETS = 1 << 20


def static_options(args):
    return {key: getattr(args, key, default) for key, default in STATIC_DEFAULTS.items()}


def add_static_arguments(parser):
    for key, default in STATIC_DEFAULTS.items():
        parser.add_argument('--' + key, type=type(default), default=os.getenv(key.upper(), default))


def static_ablation_label(options):
    mode = options['opd_static_correction']
    if mode == 'none':
        return ''
    label = (f'token_freq_a{options["opd_static_alpha"]}_{options["opd_static_count_mode"]}'
             if mode == 'token_freq' else
             f'context_cache_n{options["opd_cache_order"]}_m{options["opd_cache_slots"]}_b{options["opd_cache_bonus"]}')
    return label.replace('.', 'p').replace('+', '')


def validate_static_correction(method, interval, options):
    mode = options['opd_static_correction']
    if mode not in STATIC_MODES:
        raise ValueError(f'invalid opd_static_correction: {mode}')
    if mode != 'none' and (method != 'opd_reflex' or interval != 0):
        raise ValueError('static correction requires method=opd_reflex and opd_update_interval_rounds=0')
    for name in ('opd_static_alpha', 'opd_cache_bonus'):
        value = options[name]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if options['opd_static_count_mode'] not in ('miss', 'all'):
        raise ValueError('opd_static_count_mode must be miss or all')
    if options['opd_cache_order'] not in (1, 2):
        raise ValueError('opd_cache_order must be 1 or 2')
    if options['opd_cache_slots'] < 1:
        raise ValueError('opd_cache_slots must be positive')


def miss_tokens(path, tree):
    """Last emitted token and mask for an expanded node without a matching child."""
    last = (path.lengths - 1).clamp_min(0)[:, None]
    token = path.tokens.gather(1, last)
    node = path.packed_indices.gather(1, last)
    expanded = path.feedback_contexts.gather(1, last) >= 0
    # EOS and explicit length truncation can stop traversal despite a match.
    children = (tree.parents == node) & (tree.tokens == token)
    return token, expanded & (path.lengths[:, None] > 0) & ~children.any(1, keepdim=True)


class StaticCorrection:
    def __init__(self, vocab, num_groups, device, initial_rows, *,
                 opd_static_correction, opd_static_alpha, opd_static_count_mode,
                 opd_cache_order, opd_cache_slots, opd_cache_bonus, history=None):
        self.mode = opd_static_correction
        self.alpha = opd_static_alpha
        self.count_mode = opd_static_count_mode
        self.vocab = vocab
        self.order = opd_cache_order
        self.slots = opd_cache_slots
        self.bonus = opd_cache_bonus
        self.history = history
        self.totals = torch.zeros(4, dtype=torch.long, device=device)
        if self.mode == 'token_freq':
            self.counts = torch.zeros((num_groups, vocab), dtype=torch.int32, device=device)
            self.bias = None
        else:
            self.bucket_count = vocab if self.order == 1 else CACHE_BUCKETS
            self.table_size = num_groups * self.bucket_count
            # Inactive writers get private sink buckets, so index_copy_ has unique
            # destinations without compacting a device mask into a dynamic shape.
            self.values = torch.full((self.table_size + initial_rows, self.slots),
                                     -1, dtype=torch.int32, device=device)
            self.keys = (torch.full((self.table_size + initial_rows,), -1, dtype=torch.long,
                                    device=device) if self.order == 2 else None)
            self.sinks = self.table_size + torch.arange(initial_rows, device=device)
            self.slot_priority = torch.arange(self.slots, 0, -1, device=device)

    def _bucket(self, groups, last, previous):
        if self.order == 1:
            key = last
            offset = last.clamp_min(0)
        else:
            key = torch.where((last >= 0) & (previous >= 0), previous * self.vocab + last, -1)
            # Mix both halves of the complete key. The hash is just an address.
            offset = ((key ^ (key >> 20)) * 0x9E3779B1) & (self.bucket_count - 1)
        return key, groups * self.bucket_count + offset

    def apply(self, logits, groups, last=None, previous=None):
        if self.mode == 'token_freq':
            if self.bias is not None:
                logits.add_(self.bias.to(logits.dtype).index_select(0, groups)[:, None, :])
            return
        if last is None:
            last = self.history[:, 1:2]
            previous = self.history[:, :1]
        key, bucket = self._bucket(groups[:, None], last, previous)
        tokens = self.values[bucket]
        hit = (key >= 0) & (tokens[..., 0] >= 0)
        if self.keys is not None:
            hit &= self.keys[bucket] == key
        valid = hit[..., None] & (tokens >= 0)
        logits.scatter_add_(-1, tokens.long().clamp_min(0), valid.to(logits.dtype) * self.bonus)
        self.totals[1].add_((key >= 0).sum())
        self.totals[2].add_(hit.sum())

    def observe(self, path, root_tokens, groups, owners, tree=None):
        slots = torch.arange(path.tokens.shape[1], device=path.tokens.device)
        valid = slots[None, :] < path.lengths[:, None]
        if self.mode == 'token_freq':
            tokens = path.tokens
            if self.count_mode == 'miss':
                tokens, valid = miss_tokens(path, tree)
            indices = groups[:, None] * self.vocab + tokens.clamp_min(0)
            self.counts.view(-1).scatter_add_(0, indices.reshape(-1), valid.flatten().int())
            self.bias = (self.counts.float().log1p() * self.alpha).masked_fill_(self.counts == 0, 0.)
            self.totals[0].add_(valid.sum())
            return
        sequence = torch.cat((self.history[:, :1], root_tokens, path.tokens), dim=1)
        keys, buckets = self._bucket(groups[:, None], sequence[:, 1:-1], sequence[:, :-2])
        for position in range(path.tokens.shape[1]):
            key = keys[:, position]
            self._insert_position(buckets[:, position], key, path.tokens[:, position],
                                  owners, valid[:, position] & (key >= 0))
        last = (path.lengths + 1)[:, None]
        self.history = sequence.gather(1, torch.cat((last - 1, last), dim=1))

    def _insert_position(self, bucket, key, tokens, owners, valid):
        """Apply one time slice in increasing original response order on device."""
        old_values = self.values[bucket]
        old_key = (self.keys[bucket] if self.keys is not None
                   else torch.where(old_values[:, 0] >= 0, key, -1))
        same_bucket = (bucket[:, None] == bucket) & valid[:, None] & valid
        priority = owners[None, :].expand_as(same_bucket)
        earlier = same_bucket & (priority < owners[:, None])
        later = same_bucket & (priority > owners[:, None])
        previous_owner, previous_row = priority.masked_fill(~earlier, -1).max(dim=1)
        previous_key = torch.where(previous_owner >= 0, key[previous_row], old_key)
        self.totals[0].add_(valid.sum())
        self.totals[3].add_((valid & (previous_key >= 0) & (previous_key != key)).sum())

        same_key = key[:, None] == key
        last_conflict = priority.masked_fill(~(earlier & ~same_key), -1).amax(1)
        superseded = (later & same_key & (tokens[:, None] == tokens)).any(1)
        # The surviving bucket contains only events after the last key change.
        incoming = same_bucket & same_key & (priority > last_conflict[:, None]) & ~superseded
        existing = (old_key == key)[:, None] & (last_conflict < 0)[:, None] & (old_values >= 0)
        existing &= ~(incoming[:, :, None] & (tokens[None, :, None] == old_values[:, None, :])).any(1)
        priorities = torch.cat(((priority + self.slots + 1).masked_fill(~incoming, -1),
                                self.slot_priority.expand_as(old_values).masked_fill(~existing, -1)), dim=1)
        recency, order = priorities.topk(self.slots, dim=1)
        candidates = torch.cat((tokens.int().expand_as(same_bucket), old_values), dim=1)
        values = candidates.gather(1, order).masked_fill(recency < 0, -1)
        dest = torch.where(valid & ~later.any(1), bucket, self.sinks[:len(bucket)])
        self.values.index_copy_(0, dest, values)
        if self.keys is not None:
            self.keys.index_copy_(0, dest, key)

    def compact(self, keep):
        if self.history is not None:
            self.history = self.history.index_select(0, keep)

    def stats(self):
        # Called once after the rollout. Apply and observe never read device scalars.
        nonzero = (self.counts.count_nonzero() if self.mode == 'token_freq'
                   else self.totals.new_zeros(()))
        observations, lookups, hits, collisions, nonzero = torch.cat(
            (self.totals, nonzero.reshape(1))).tolist()
        return dict(opd_static_observations=observations, opd_static_nonzero_counts=nonzero,
                    opd_static_cache_lookups=lookups, opd_static_cache_hits=hits,
                    opd_static_cache_collisions=collisions,
                    opd_static_cache_buckets=self.bucket_count if self.mode == 'context_cache' else 0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--interval', type=int, required=True)
    add_static_arguments(parser)
    args = parser.parse_args()
    options = static_options(args)
    try:
        validate_static_correction(args.method, args.interval, options)
    except ValueError as error:
        parser.error(str(error))
    print(static_ablation_label(options))
