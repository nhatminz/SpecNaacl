# Implementation report

## Result

`SpecNaacl/` was created as an independent sibling copy; `fastgrpo/` was not
modified. The rollout now supports `REFLEX_MODE=off|active`. The disabled path
retains the original proposal softmax and FastGRPO verifier. Active mode uses a
compact-vocabulary, trajectory-local `FastLKReflex`, updates only at the root,
and reuses the verification target logits.

No long training and no fabricated experimental output were produced.

## Main files created or modified

- `helper/fast_lk_reflex.py`: fixed projection, per-response A state, analytic
  LK gradient/update, active-batch removal and aggregate counters.
- `helper/specualtive_generate.py`: compact EAGLE proposal correction and
  root feedback hook; FastGRPO tree verification is unchanged.
- `helper/eagle3_specforge.py`: minimal native compact-logit and fixed
  compact-to-target vocabulary adapter.
- `helper/checkpointing.py`, `grpo_speculative.py`: RNG/optimizer/gradient
  accumulation resume, synchronous gradient all-reduce, rank-0 tqdm/logging,
  Reflex CLI/config/metrics, and lightweight timing CSV.
- `helper/drift_metrics.py`: removes the accidental dependency on the sibling
  MEDUSA project. It is only a compatibility metric for the legacy draft path.
- `scripts/launch/{pretrain_model,train_model}.sh`: shared launch logic.
- Seven `pretrain_<model>.sh` and seven `train_<model>.sh` wrappers for Qwen2.5
  1.5B/3B/7B/14B, Qwen3 1.7B/4B, and Llama 3.1 8B.
- `scripts/generate_eagle3_config.py`: explicit Qwen2, Qwen3, and Llama family
  validation; unknown families fail rather than fall back.
- `scripts/prepare_local_pretrain_data.py`, `scripts/write_run_metadata.py`.
- `scripts/plot_training_time.py` and `.sh`.
- `README.md`, `RUNNING.md`, `METHOD_FAST_LK_REFLEX.md`, `huongdanchay.md`.
- `tests/test_fast_lk_reflex.py`, `tests/test_checkpointing.py`,
  `tests/test_model_configs.py`, `tests/test_shell_scripts.py`, and `pytest.ini`.

The vendored SpecForge source and previously implemented Torch 2.11 /
Transformers 5.8.1 / SGLang 0.5.14 compatibility fixes remain in
`third_party/SpecForge`. Provenance is recorded in `VENDORED_COMMIT` and
`DEPENDENCIES_POLICY_LAG.md`.

## Preserved versus changed

Preserved from FastGRPO: GRPO objective and rewards, policy update ordering,
adaptive concurrency/tree budget, target verification semantics, accepted
length counters, and persistent online draft updates.

Changed: EAGLE-3 proposal logits may receive the temporary `A psi` correction;
after verification the analytic LK root feedback updates only A. The existing
SpecForge `OnlineEagle3Model` still owns persistent EAGLE architecture, feature
projection, loss and training-time unrolling. Reflex has no optimizer and does
not enter a checkpoint because it ends with each completed rollout.

## Validation performed on this workstation

- `python3 -m compileall -q .`: passed, including vendored source.
- `bash -n` over every shell script: passed.
- Dry-run of all 14 model wrappers: passed.
- Model config generation tests for Qwen2, Qwen3 and Llama plus unsupported
  family rejection: passed.
- Available CPU test subset: `5 passed`.

The complete `pytest -q` suite was collected, but this workstation has no
PyTorch installation (`ModuleNotFoundError: torch`), so tensor/autograd,
optimizer/RNG and state-removal tests could not execute here. The tests are
present and should be run in the B200 Python environment.

Not verified locally because this machine has no B200/model checkpoints and no
matching CUDA stack: real collect -> persistent SpecForge draft train -> target
GRPO train, multi-GPU NCCL, and end-to-end active/off acceptance throughput.
Run `python -m pytest -q` before the first long job, then use a small
`MAX_TRAIN_SAMPLES`/`GEN_MAX_LENGTH` smoke run. No full training is launched by
the setup scripts automatically.
