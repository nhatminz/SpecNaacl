# Implementation report

## Files modified

- `helper/specualtive_generate.py`
- `helper/fast_lk_reflex.py`
- `helper/sampling.py` (new)
- `grpo_speculative.py`
- `tests/test_fast_lk_reflex.py`
- `tests/test_sampling.py` (new)
- `pytest.ini`
- `README.md`
- `RUNNING.md`
- `METHOD_FAST_LK_REFLEX.md`
- `IMPLEMENTATION_REPORT.md`

## Bugs fixed

- OFF and ACTIVE now share the same FP32 compact EAGLE softmax, top-k and
  fixed `d2t` mapping. ACTIVE differs only by adding `A psi`; a zero A parity
  test covers probabilities, compact top-k ids and mapped target ids.
- Target sampling and Reflex supervision reuse one normalized probability
  tensor after temperature, top-p and top-k. Reflex adds no target forward or
  separate full-vocabulary softmax.
- LK now conditions and renormalizes target probability on the compact
  vocabulary. Analytic-gradient, nonzero-gradient and descent tests were added.
- The fast-state update uses in-place batched `baddbmm`, including `beta=1`
  when weight decay is zero, and finished trajectories are compacted once per
  verification round.
- Pytest resolves repository modules without manually setting `PYTHONPATH`.
- Log/final AAL, acceptance, rollout-token, reward and loss counters are reduced
  across ranks. Timing uses the maximum rank time; throughput uses global
  cumulative tokens divided by cumulative elapsed job time.
- Checkpoint v3 gathers per-rank Python, NumPy, Torch CPU and CUDA RNG, rank-local
  accumulated gradients and buffers; it records world size and cumulative
  elapsed time. Resume selects the current rank state and rejects world-size
  mismatch. Both optimizer states and accumulated gradients remain restorable.
- Default Reflex profiling remains disabled and no per-token/per-round disk log,
  diagnostic top-k, or default-path synchronization was added.

## Test commands

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
python -m compileall -q .
pytest -q
find . -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

## Test result on this workstation

- `python3 -m compileall -q SpecNaacl`: passed.
- Shell syntax check for every `.sh` under `SpecNaacl`: passed.
- Non-Torch tests: `5 passed`.
- Full `pytest -q`: collection stopped because this workstation does not have
  PyTorch (`ModuleNotFoundError: torch`) for three tensor test modules. Tests
  were not skipped or weakened to hide the missing dependency.

## Not verified without B200/model checkpoints

- Full tensor/autograd suite in the pinned B200 environment.
- Real OFF/ACTIVE EAGLE-3 rollout parity with model checkpoints.
- End-to-end FastGRPO training, NCCL multi-GPU checkpoint/resume, and measured
  B200 wall-clock/throughput. No full training or speedup benchmark was run.
