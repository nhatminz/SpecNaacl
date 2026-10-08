# ReflexOPD sampler: trước / sau

- GPU→CPU sync đã bỏ ở finite path: ba scalar checks (`isnan.any`,
  `isinf.any`, `valid_mask.any`) và dynamic `where(mask)[0]`/nonzero.
  Profiler CUDA stream synchronization: **4 → 0**. Softmax, sort và
  multinomial vẫn mỗi loại đúng một lần với top-p=.95.
- Parity: **151 sampler tests pass**, toàn bộ suite **324 pass**.
  Token IDs, probabilities, CPU/CUDA RNG và sort/teacher metadata bitwise;
  temperature/top-p/top-k, BF16/FP16/FP32, nhiều shapes/seeds, ties,
  strided logits, strict NaN/Inf/all-invalid/mixed fallback đã pass.
  Teacher feedback/A/B gradients và rollout acceptance/forward parity pass.
  CUDA assertion từ chối nonfinite input; finite path capture test cũng pass.
- Sampler latency sau warmup, RTX 3090, BF16, V=151936, temperature=1,
  top-p=.95, top-k disabled. Median wall latency, 5 paired rounds,
  10 calls/round, 5 warmup calls/mode:

  | Rows | Trước (ms) | Sau (ms) |
  |---:|---:|---:|
  | 1 | 0.5618 | 0.3744 |
  | 64 | 3.3514 | 3.0783 |
  | 512 | 19.1507 | 17.8551 |

  **B200 chưa đo vì máy hiện có RTX 3090. Mặc định vẫn `strict`.**
  Benchmark B200 dùng [benchmark_opd_sampler.py](scripts/benchmark_opd_sampler.py)
  với `--require-b200`, target/draft/dataset thật và workload cần đo.
  Finite path chỉ opt-in bằng `OPD_SAMPLER_MODE=finite` trước startup;
  assertion trên GPU kiểm tra finite logits, strict mode giữ fallback gốc.
- End-to-end generation: **72.40 → 72.81 tokens/s** trên RTX 3090,
  cùng Qwen2.5-1.5B-Instruct, cùng draft checkpoint 2 pretrain steps,
  SimpleLR, batch=1, responses=2, max_length=128, max_prompt=96,
  K=2/depth=3, stream=1, seeds=42/43. Median 5 paired rounds,
  mỗi round 4 rollouts; warmup toàn bộ schedule cho cả hai mode.
  Tokens/RNG/acceptance/forward counts khớp. Đây là smoke workload;
  chưa có số đo end-to-end trên B200.

Raw reports và log: [validation artifacts](validation/opd_sampler_20261008/).
