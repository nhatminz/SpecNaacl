# OPD: static KV, sampler reuse, selected-state feedback, iteration logs

Phạm vi: chỉ OPD generation và telemetry chung. Không đổi historical FastGRPO
generation, reward, target sampler/RNG, verifier, GRPO loss hay EAGLE training
schedule. Không đổi dependencies hoặc các đường dẫn model/data/pretrained draft.

## Những thay đổi đã implement

- `helper/opd_static_cache.py`: Cache API của Transformers 5.12.1, fixed-capacity
  K/V pools cho target và EAGLE draft. Append ghi suffix; rollback/crop đổi length;
  attention chỉ nhìn prefix hợp lệ, không attention vào toàn capacity. EAGLE tree
  expansion rollback về prefix ban đầu, không làm committed cache dài thêm.
  Accepted target suffix vẫn gather vào scratch nhỏ rồi copy in-place.
- `helper/opd_sampling.py`: cùng FP32 temperature, softmax, top-p/top-k operations
  và multinomial như sampler cũ; giữ lại sorted IDs/probabilities đã có. Full
  vocab/permutation lấy teacher Top16 từ sampler prefix, compact_mass=1; greedy
  lấy trực tiếp sampled ID. Không thêm target hoặc draft transformer forward.
- Positive ties ở biên teacher được merge bằng integer probability/compact-ID key,
  giữ low-compact-ID tie rule; token p=0 không hợp lệ. Không sort/softmax lại vocab.
  **Ngoại lệ cần nói rõ:** khi positive boundary tie rất lớn, phải đọc phần tied
  suffix để giữ exact tie rule; trường hợp uniform có thể đọc cả sorted support.
  Với compact subset hoặc sampling không tạo sorted intermediate, vẫn scan compact
  probabilities của **selected states**, không phải mọi verification row.
- `helper/opd_reflex{,_kernels}.py`: sparse scores [contexts, capacity-S], active
  slot mapping [V]; không reserve dense [contexts,V] cho sparse/fused-dense. S reserve
  gồm số active đã biết + bound cho một feedback update chưa phản ánh vào host packet;
  tăng capacity theo cấp số nhân, không allocate mỗi round. Dense GEMM workspace
  chỉ tạo khi thực sự được chọn và reuse. Dispatch chỉ launch một backend.
- Teacher/head/union/B update/A terms xử lý GPU-selected work queue bằng persistent
  workers, không launch K*all-tree-rows rồi skip. A reduction chỉ đọc selected rows.
  Union/dedup/p/q/tail/KL/q-p vẫn fused như trước; không thêm heuristic gating.
- `helper/specualtive_generate.py`, `helper/tree_verification.py`: reuse candidate,
  parent/context/token/position/confidence pools, branch TopK output, ping-pong
  ancestry indices, draft attention mask và packed-tree buffers. Bỏ Python history
  lists và các concat ở tree expansion. Native small-tree TopK/gather vẫn có
  internal scratch; không claim zero allocations cho toàn decoder.
- `helper/opd_optimizer.py`: optional separate A LR; giữ AdamW moments khi migrate
  checkpoint single-group sang two-group. Resume two-group khi không set LR mới
  vẫn giữ LR đã lưu. A persistent; B reset mỗi rollout; không optimizer/backward/
  all-reduce mỗi round. Unset LR giữ setup cũ.

## Đồng bộ và memory: giới hạn thực tế

Scheduling host sync/verification round: **1 trước, 1 sau** (tiny D2H packet).
Không thêm scalar GPU read để chọn backend, chọn teacher hoặc ghi CSV. Update stream
vẫn wait ngay trước proposal tiếp theo; cả stream=0/1 có parity tests.
Prefill và natural end-rollout vẫn có transfers. Optional profiling có timing sync
riêng; production PROFILE=DIAGNOSTICS=STATISTICAL_TIME=0 không thêm sync.

KV không copy full prefix **khi append/crop/accept mỗi round**. Prefill repeat và
finished-response batch compaction vẫn copy các prefix đang sống; không claim đã
loại các exceptional copies này. Static pool dùng conservative padded-length bound:
prompt_width + max_length*(max_depth+1) + verification_limit + max_k*max_depth.
Đây là tradeoff VRAM để tránh grow/cat; B200 peak VRAM với model lớn chưa đo.
Thay layout KV và reduction grad_A giữ cùng toán học, nhưng không hứa bitwise
trajectory trên mọi GPU/attention kernel hoặc qua các checkpoint revision.

## CSV theo DataLoader iteration

`outputs/train/<model_key>/<run_name>/logs/rollout_timing.csv`:
đúng một row cho mỗi iteration được thực thi, kể cả reward std=0, prompt quá dài,
answer None hoặc GRPO step chưa tăng. Resume bỏ qua các iteration đã checkpoint,
không ghi trùng. `global_iter` liên tục qua epochs/resume; resume checkpoint cũ
không có telemetry dùng iterator position và cumulative counters cũ.

- iter_aal = total_acc_length / total_decoded_token_num của riêng rollout.
- cumulative_aal = tổng accepted lengths / tổng sequence verification rounds.
- Accepted-length definition kế thừa FastGRPO, gồm target-sampled root/bonus
  theo committed path; không average các batch AAL ratios.
- Acceptance rate dùng accepted_draft / proposed_draft, khác AAL.
- Wall time từ lúc Python training process bắt đầu đến cuối iteration, gồm
  model/setup, generation, reward, training và checkpoint work đã hoàn thành;
  resume cộng elapsed đã lưu. Downtime giữa các phiên không tính.
- OPD fields dùng host counters ở natural rollout boundary. Không giữ GPU history
  trong telemetry. File handle buffered, flush mỗi ROLLOUT_LOG_FLUSH_INTERVAL,
  sau checkpoint, exception/KeyboardInterrupt, SIGTERM và kết thúc.
- Multi-rank: mỗi rank có file riêng (`rollout_timing.rankN.csv`); file không suffix
  là rank0, không phải all-rank sum. Không thêm all-reduce chỉ để log.
- `timing.csv` và `metrics.jsonl` per-GRPO-step vẫn giữ.

## B200: lệnh chạy

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0
export MODEL_KEY=qwen3_1p7b

# Dùng config pretrained thật của model; tự đọc compact vocab, hidden size.
bash scripts/tune_opd_proposals.sh

# Đo frozen-rollout với historical FastGRPO + OPD, response dài, 2 stream modes.
# Không dùng kết quả timing từ diagnostic replay để claim speedup.
OPD_FAST_LRS=0.001,0.01,0.05 OPD_STREAMS=0,1 BENCH_SEEDS=42,43 \
BENCH_ITERATIONS=2 BENCH_MAX_LENGTH=2048 BENCH_MAX_PROMPT_LENGTH=256 \
bash scripts/sweep_opd_reflex.sh --gpu-utilization

# Chọn LR/stream sau khi xem report.json; dưới đây vẫn là defaults chưa tune.
OPD_PROJECTOR_LR=1e-5 OPD_FAST_LR=0.01 OPD_UPDATE_STREAM=0 \
ROLLOUT_LOG_FLUSH_INTERVAL=8 bash train_qwen3_1p7b.sh
bash train_qwen3_1p7b_fastgrpo.sh
```

Đổi MODEL_KEY và launcher sang `qwen25_3b` hoặc `qwen3_4b` tương ứng.
Tất cả 14 train wrappers vẫn có cặp OPD/FastGRPO và env overrides.
Defaults giữ: SDPA, target/draft LR 1e-5, batch8, accumulation4, responses8,
draft token budget2048, OPD rank8/Top16, fastLR0.01, update_stream0, profile0,
diagnostics0, proposal/dense implementation auto.

## Validation và benchmark

Xem section đầu IMPLEMENTATION_REPORT.md. Máy kiểm tra là RTX3090, không phải
B200. Production model/data/pretrained checkpoint không có tại máy này; chưa có
FastGRPO-vs-OPD AAL/tokens/s hoặc B200 crossover thực. Không tự launch full training.
Tuning component JSON chỉ là synthetic fixture và không được deploy sang B200.
