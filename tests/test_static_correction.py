"""Reference, integration and launcher tests for the online-correction baselines."""
from contextlib import nullcontext
from copy import deepcopy
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from helper import static_correction as correction
from helper.opd_reflex import select_states_reference
from helper.tree_verification import PackedTree, VerifiedPath, trace_verified_path


ROOT = Path(__file__).resolve().parents[1]
DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))]


class ReferenceCorrection:
    """Sequential plain Python implementation, including bucket replacement."""
    def __init__(self, mode, vocab, groups, history, count_mode='miss', order=1, slots=1):
        self.mode = mode
        self.vocab = vocab
        self.groups = groups
        self.history = deepcopy(history)
        self.count_mode = count_mode
        self.order = order
        self.slots = slots
        self.counts = {}
        self.buckets = {}
        self.observations = self.lookups = self.hits = self.collisions = 0

    def address(self, group, key):
        if self.order == 1:
            return group, key[0]
        complete = key[0] * self.vocab + key[1]
        return group, ((complete ^ (complete >> 20)) * 0x9E3779B1) % correction.CACHE_BUCKETS

    def successors(self, group, key):
        if min(key) < 0:
            return []
        self.lookups += 1
        occupied, values = self.buckets.get(self.address(group, key), (None, []))
        if occupied != key:
            return []
        self.hits += 1
        return values

    def observe(self, tokens, lengths, contexts, indices, parents, tree_tokens, groups, owners):
        import collections

        if self.mode == 'token_freq':
            for row, width in enumerate(lengths):
                group = groups[row]
                if width == 0:
                    continue
                selected = tokens[row][:width]
                if self.count_mode == 'miss':
                    last = width - 1
                    child = (indices[row][last], selected[-1])
                    if contexts[row][last] < 0 or child in list(zip(parents[row], tree_tokens[row])):
                        continue
                    selected = selected[-1:]
                counts = self.counts.setdefault(group, collections.Counter())
                counts.update(selected)
                self.observations += len(selected)
            return

        ordered_rows = sorted(range(len(lengths)), key=owners.__getitem__)
        for position in range(max(lengths, default=0)):
            for row in ordered_rows:
                if position >= lengths[row]:
                    continue
                key = tuple(self.history[row][-self.order:])
                token = tokens[row][position]
                if min(key) >= 0:
                    address = self.address(groups[row], key)
                    occupied, values = self.buckets.get(address, (None, []))
                    if occupied != key:
                        values = []
                        self.collisions += occupied is not None
                    self.buckets[address] = key, ([token] + [old for old in values if old != token])[:self.slots]
                    self.observations += 1
                self.history[row] = [self.history[row][-1], token]

    def compact(self, keep):
        self.history = [self.history[row] for row in keep]


def make_path(tokens, lengths, contexts, indices, device):
    return VerifiedPath(*(torch.tensor(value, device=device) for value in (tokens, indices, contexts, lengths)))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('mode,option', [('token_freq', 'all'), ('token_freq', 'miss'),
                                         ('context_cache', 1), ('context_cache', 2)])
@pytest.mark.parametrize('grouped', [False, True])
def test_reference_rounds_compaction_and_logits(device, mode, option, grouped, monkeypatch):
    import math

    # Small table to exercise collisions in a short test.
    monkeypatch.setattr(correction, 'CACHE_BUCKETS', 16)
    vocab = 17
    options = correction.STATIC_DEFAULTS | dict(opd_static_correction=mode, opd_static_alpha=1.7,
        opd_cache_bonus=.7, opd_cache_order=option if mode == 'context_cache' else 1,
        opd_static_count_mode=option if mode == 'token_freq' else 'miss', opd_cache_slots=4)
    owners = [3, 0, 2, 1, 4]
    groups = [owner % 2 if grouped else 0 for owner in owners]
    history = [[0, 1], [1, 1], [1, 1], [0, 1], [2, 1]]
    state = correction.StaticCorrection(vocab, 2 if grouped else 1, device, 5,
        history=torch.tensor(history, device=device), **options)
    ref = ReferenceCorrection(mode, vocab, groups, history, options['opd_static_count_mode'],
                              options['opd_cache_order'], options['opd_cache_slots'])
    real_rows = [[4,1,5,8], [5,7,4,4], [4,7,1,8], [2,8,1,2], [3,2,3,5]]
    for lengths, keep in [([4, 1, 4, 0, 1], [4, 2, 0]), ([1, 4, 3], [1, 2]), ([4, 4], [])]:
        group_tensor = torch.tensor(groups, device=device)
        roots = torch.tensor([[row[-1]] for row in ref.history], device=device)
        tokens = [row[:length] + [-1] * (4 - length) for row, length in zip(real_rows, lengths)]
        indices = [list(range(length)) + [-1] * (4 - length) for length in lengths]
        contexts = [[0, 1, 2 if row % 2 else -1, 3][:length] + [-1] * (4 - length)
                    for row, length in enumerate(lengths)]
        parents = [[-1, 0, 1, 2, 0] for _ in groups]
        tree_tokens = [[root[-1], 2, 8, 4, 5] for root in ref.history]
        tree = PackedTree(torch.tensor(parents, device=device), torch.tensor(tree_tokens, device=device),
                          torch.tensor([[0, 1, 2, 3, -1]] * len(groups), device=device), 3)
        state.observe(make_path(tokens, lengths, contexts, indices, device), roots, group_tensor,
                      torch.tensor(owners, device=device), tree)
        ref.observe(tokens, lengths, contexts, indices, parents, tree_tokens, groups, owners)
        if mode == 'token_freq':
            assert state.counts.tolist() == [
                [ref.counts.get(group, {}).get(token, 0) for token in range(vocab)]
                for group in range(2 if grouped else 1)]
        else:
            assert state.history.tolist() == ref.history
            actual = {(group, index): row[:next((i for i,v in enumerate(row) if v < 0), len(row))]
                      for group in range(2 if grouped else 1)
                      for index, row in enumerate(state.values[group * state.bucket_count:
                                                     (group + 1) * state.bucket_count].tolist()) if row[0] >= 0}
            assert actual == {address: vals for address, (_, vals) in ref.buckets.items()}
            if state.keys is not None:
                for (group, address), (key, _) in ref.buckets.items():
                    assert state.keys[group * state.bucket_count + address] == key[0] * vocab + key[1]

        # Root, a valid expansion, and a missing key or group are checked in bf16.
        for expansion in (False, True):
            root_keys = [ref.history[row][-2:] for row in range(len(groups))]
            keys = [[[previous, last], [last, 4], [-1, -1]] if expansion else [[previous, last]]
                    for previous, last in root_keys]
            logits = torch.zeros((len(groups), len(keys[0]), vocab), device=device, dtype=torch.bfloat16)
            expected = torch.zeros_like(logits)
            for row, group in enumerate(groups):
                for column, key in enumerate(keys[row]):
                    if mode == 'token_freq':
                        for token, count in ref.counts.get(group, {}).items():
                            expected[row, column, token] = options['opd_static_alpha'] * math.log1p(count)
                    else:
                        for token in ref.successors(group, tuple(key[-ref.order:])):
                            expected[row, column, token] = options['opd_cache_bonus']
            keys = torch.tensor(keys, device=device)
            state.apply(logits, group_tensor, keys[..., 1] if expansion else None,
                        keys[..., 0] if expansion else None)
            assert torch.equal(logits, expected)
        actual = state.stats()
        assert actual['opd_static_observations'] == ref.observations
        assert actual['opd_static_cache_collisions'] == ref.collisions
        assert actual['opd_static_cache_lookups'] == ref.lookups
        assert actual['opd_static_cache_hits'] == ref.hits
        state.compact(torch.tensor(keep, device=device, dtype=torch.long))
        ref.compact(keep)
        groups = [groups[row] for row in keep]
        owners = [owners[row] for row in keep]
        real_rows = [real_rows[row] for row in keep]


def test_misses_require_an_expanded_unmatched_node_even_at_eos_or_cap():
    parents = torch.tensor([[-1, 0, 0, 1]] * 3)
    tree = PackedTree(parents, torch.tensor([[8, 1, 9, 4]] * 3),
                      torch.tensor([[0, 1, -1, -1]] * 3), 2)
    path = trace_verified_path(tree, torch.tensor([[1, 7, 2, 3], [9, 7, 2, 3], [1, 4, 2, 6]]), 9)
    tokens, miss = correction.miss_tokens(path, tree)
    assert tokens.flatten().tolist() == [7, 9, 6]
    assert miss.flatten().tolist() == [True, False, False]
    _, kind = select_states_reference(tree, path)
    assert kind[0, path.packed_indices[0, path.lengths[0] - 1]] == 1
    path.lengths[0] = 1
    assert not correction.miss_tokens(path, tree)[1].any()


def tiny_model(device):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from helper.fastgrpo_model import FastGRPOModel

    torch.manual_seed(321)
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    config = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=128, attention_dropout=0., torch_dtype=dtype)
    config._attn_implementation = 'sdpa'
    target = Qwen2ForCausalLM(config).to(device=device, dtype=dtype).eval()
    draft = deepcopy(config)
    draft.num_hidden_layers = 1
    model = FastGRPOModel(draft, target).to(device).eval()
    model.enable_opd(8)
    return model


def tiny_generate(generate, model, interval, **kwargs):
    torch.manual_seed(715)
    return generate(model, torch.tensor([[0,7,9], [3,5,8]]), torch.tensor([[0,1,1], [1,1,1]]),
        SimpleNamespace(eos_token_id=96), do_sample=kwargs.pop('do_sample', True),
        method=kwargs.pop('method', 'opd_reflex'), repeated_generate_nums=2, temperature=.8,
        top_p=.95, max_length=14, verification_capacity=28, max_verification_num=7, max_draft_k=2,
        max_draft_token_length=3, min_draft_token_length=3, opd_b_sharing_scope='per_prompt_group',
        opd_update_interval_rounds=interval, **kwargs)


@pytest.fixture(params=DEVICES)
def tiny_device(request, monkeypatch):
    limit = torch.get_num_threads()
    torch.set_num_threads(1)
    if request.param == 'cpu':
        # The CPU oracle tests fp32 feedback. CUDA exercises the bf16 path.
        monkeypatch.setattr(torch.amp, 'autocast', lambda *args, **kwargs: nullcontext())
    try:
        yield request.param
    finally:
        torch.set_num_threads(limit)


@pytest.mark.parametrize('interval', [0, 1])
def test_off_matches_clean_checkout(tiny_device, interval):
    from helper.opd_generate import speculative_generate

    try:
        source = subprocess.check_output(['git', 'show', 'b158557:helper/opd_generate.py'],
                                         cwd=ROOT, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pytest.skip('b158557 is required for the golden comparison')
    clean = {}
    exec(compile(source, 'b158557:helper/opd_generate.py', 'exec'), clean)
    expected = tiny_generate(clean['speculative_generate'], tiny_model(tiny_device), interval)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if tiny_device == 'cuda' else None
    actual = tiny_generate(speculative_generate, tiny_model(tiny_device), interval, opd_static_correction='none')
    stable = lambda row: {key: value for key, value in row.items()
                          if 'time' not in key and not key.endswith('_ms')}
    assert stable(actual) == stable(expected)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)


@pytest.mark.parametrize('mode,opts', [('token_freq', dict(opd_static_count_mode='all')),
                                       ('context_cache', dict(opd_cache_order=2, opd_cache_slots=4))])
def test_greedy_tokens_hooks_and_reset(tiny_device, mode, opts, monkeypatch):
    import helper.opd_generate as generation
    from helper.opd_reflex import OPDReflex

    model = tiny_model(tiny_device)
    expected = tiny_generate(generation.speculative_generate, model, 0, do_sample=False, max_response_tokens=10)
    snapshots = []
    class SpyCorrection(correction.StaticCorrection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.proposals = []
            snapshots.append(self.proposals)
            assert self.stats()['opd_static_observations'] == 0

        def apply(self, logits, groups, last=None, previous=None):
            from math import log1p
            raw = logits.clone()
            super().apply(logits, groups, last, previous)
            expected_logits = raw.clone()
            for row, group in enumerate(groups.tolist()):
                for col in range(logits.shape[1]):
                    if self.mode == 'token_freq':
                        for token, count in enumerate(self.counts[group].tolist()):
                            value = torch.tensor(self.alpha * log1p(count), dtype=logits.dtype, device=logits.device)
                            expected_logits[row, col, token] += value
                    else:
                        key = self.history[row].tolist() if last is None else [
                            int(previous[row, 0 if previous.shape[1] == 1 else col]), int(last[row, col])]
                        addr = ReferenceCorrection(self.mode, self.vocab, [], [], order=2).address(group, key)
                        bucket = addr[0] * self.bucket_count + addr[1]
                        if int(self.keys[bucket]) == key[0] * self.vocab + key[1]:
                            for token in self.values[bucket].tolist():
                                if token >= 0:
                                    expected_logits[row, col, token] += self.bonus
            assert torch.equal(logits, expected_logits)
            self.proposals.append((last is None, logits))

    propose = OPDReflex.propose
    def spy_propose(self, logits, *args, **kwargs):
        root, corrected = snapshots[-1].pop()
        assert kwargs.get('root', False) == root
        assert corrected is logits
        return propose(self, logits, *args, **kwargs)

    monkeypatch.setattr(generation, 'StaticCorrection', SpyCorrection)
    monkeypatch.setattr(OPDReflex, 'propose', spy_propose)
    for _ in range(2):
        actual = tiny_generate(generation.speculative_generate, model, 0, do_sample=False,
                               max_response_tokens=10, opd_static_correction=mode, **opts)
        assert actual['generated_token_ids'] == expected['generated_token_ids']
        assert actual['opd_static_observations'] > 0
        assert not snapshots[-1]
    assert len(snapshots) == 2


def test_guards_and_fastgrpo_dispatch(monkeypatch):
    from helper import fastgrpo_generate, opd_generate, specualtive_generate

    monkeypatch.setattr(fastgrpo_generate, 'speculative_generate', lambda **kwargs: kwargs)
    assert specualtive_generate.speculative_generate(method='fastgrpo', opd_static_correction='token_freq',
                                                      opd_cache_slots=4, do_sample=False) == {'do_sample': False}
    model = SimpleNamespace(target_model=SimpleNamespace(device='cpu', config=SimpleNamespace()))
    with pytest.raises(ValueError, match='requires.*opd_update_interval_rounds=0'):
        opd_generate.speculative_generate(model, torch.zeros((1,1)), torch.ones((1,1)), None,
                                          opd_static_correction='token_freq')


def test_launcher_flags_paths_and_invalid_combinations(tmp_path):
    env = os.environ | dict(DRY_RUN='true', DATASET='gsm8k', PYTHON_BIN=sys.executable, OUTPUT_ROOT=str(tmp_path))

    def launch(script='train_qwen25_3b.sh', **settings):
        return subprocess.run(['bash', script], cwd=ROOT, env=env | settings, text=True, capture_output=True)

    paths = set()
    for mode, interval in [('none', '1'), ('none', '0'), ('token_freq', '0'), ('context_cache', '0')]:
        run = launch(OPD_STATIC_CORRECTION=mode, OPD_UPDATE_INTERVAL_ROUNDS=interval,
                     OPD_STATIC_ALPHA='0.5', OPD_STATIC_COUNT_MODE='all',
                     OPD_CACHE_ORDER='2', OPD_CACHE_SLOTS='4', OPD_CACHE_BONUS='8')
        assert run.returncode == 0, run.stderr
        line = next(line for line in run.stdout.splitlines() if line.startswith('Command'))
        command = shlex.split(line.split(':', 1)[1])
        for flag, value in dict(opd_static_correction=mode, opd_static_alpha='0.5',
            opd_static_count_mode='all', opd_cache_order='2', opd_cache_slots='4', opd_cache_bonus='8').items():
            assert command[command.index('--' + flag) + 1] == value
        paths.add(Path(command[command.index('--log_file') + 1]).parents[2])
    assert len(paths) == 4
    assert 'requires method=opd_reflex' in launch('train_qwen25_3b_fastgrpo.sh',
        OPD_STATIC_CORRECTION='token_freq', OPD_UPDATE_INTERVAL_ROUNDS='0').stderr
    assert 'opd_static_alpha must be positive' in launch(OPD_STATIC_CORRECTION='token_freq',
        OPD_UPDATE_INTERVAL_ROUNDS='0', OPD_STATIC_ALPHA='nan').stderr
