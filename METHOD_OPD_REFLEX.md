# ReflexOPD trên FastGRPO

Target/draft, shared embedding/lm_head, adaptive K/depth, confidence pruning,
acceptance và online objective lấy từ FastGRPO. Baseline thực thi source gốc;
OPD dùng tensor tree/verifier và KV storage tối ưu với cùng quy tắc cây.
Pruning confidence dùng native torch.topk như upstream.

A persistent có shape `[hidden_size, rank]` (mặc định rank 8), được học cùng
optimizer draft và lưu checkpoint. B_fast FP32 có shape `[full_vocab_size, rank]`,
được reset mỗi rollout. Với head-input h: `u=h@A`,
`corrected_logits=raw_logits+u@B_fast.T`; tree dùng `DraftTop16[:draft_k]`.
B=0 giữ nguyên raw logits. Top16 tie IDs/FP32 normalization có thể khác native
TopK(K); không yêu cầu OPD sinh cùng token với baseline khi proposal khác.

Teacher là xác suất sampler target thực sự dùng sau temperature/top-p/top-k.
Reuse probabilities và sort/topk có sẵn của sampler để trích TargetTop16 và
p_target tại DraftTop16; callback kết thúc trước khi arrays full `[N,V]` bị
release. Async feedback chỉ giữ metadata nhỏ. Không thêm target softmax/sort
hoặc transformer forward. Greedy teacher là delta tại sampled token.

Feedback chọn visited states và rejected siblings cách một cạnh, chỉ với
context có proposal tương ứng. Hợp `TargetTop16 ∪ DraftTop16` khử trùng lặp,
phần còn lại là tail bucket. Forward KL và gradient q-p được tính theo phân
phối coarsened này. B cập nhật với tổng state weight; gradient A cộng vào
buffer GPU và apply tại draft optimizer boundary hiện hữu.

CUDA luôn dùng Triton (CPU reference chỉ phục vụ oracle tests). Có các backend
sparse/fused/GEMM và profile dispatch; lựa chọn backend không đọc scalar GPU.
Stream async dùng snapshot active count trễ một round và upper bound scratch.
Pool KV có thể grow và tái dùng qua rollout, swap-remove batch rows, suffix
compaction tối thiểu, contiguous history và reusable attention/tree buffers.
Sampler được đưa lại thứ tự response chuẩn sau swap-remove để giữ target RNG.
Vẫn có host scheduling và các kiểm tra sampler upstream; không tuyên bố zero-sync.

Metric cũ `opd_compact_mass_sum` được giữ tên để CSV/report không vỡ; support
hiện tại là full vocabulary, không có compact vocabulary hay mapping file.
Các đại lượng chính: KL, selected/visited/frontier states, target mass trong
DraftTop16, active rows và sparse/fused/GEMM rounds. Profiling/diagnostics là
opt-in; counter GPU được lấy tại boundary có sẵn.

Xem [báo cáo kiểm thử](FASTGRPO_REWRITE.md) và [autotuning](OPD_AUTOTUNING.md).
