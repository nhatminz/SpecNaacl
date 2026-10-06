# Learned OPD Reflex / historical baseline — current revision

FastGRPO is historical c3f05ad OFF, native softmax/topk(draft_k), original
tree packing/Python verifier/target sampler/RNG/stacked KV. No OPD optimization
or additional Top16 pass is used in this baseline.

OPD: head_input = h if norm_output else draft.norm(h), u=head_input@A.
A:[H,8] is a persistent nn.Parameter ON THE DRAFT MODEL. Old checkpoints without A
initialize once from a deterministic orthonormal basis of output-head rows,
NOT a random per-rollout/fixed projector. Existing A is loaded, never reset.
Gradient terms v_s=sum_U w_s(q-p)B_t[j] and head_input.T@v accumulate GPU-only,
BEFORE B changes. Apply weighted mean at existing draft optimizer boundary
before existing DDP sync; no round backward/optimizer/all-reduce. Pending
accumulators saved for training resume. Evaluation/frozen sweep does not accumulate
A gradients. A must actually receive optimizer steps to call it trained;
head-basis initialization alone is NOT evidence of learned improvement.

B:[compactV,r] is ONE shared rollout-local FP32 adapter. Visited + expanded,
final-packed one-hop rejected sibling states, no recursive rejected descendants.
Teacher uses existing post-temperature/top-p/top-k probabilities conditioned on
compact mass. Zero/nonfinite compact mass skipped with counter. Target candidate
entry is VALID only if p>0; unused Top16 slots use sentinel-1/prob0 and NEVER add
arbitrary q-0 rows. DraftTop16 remains included even when its teacher p is0.
Union + one tail KL, no union renormalization; q-p exact union-coordinate
gradient with support fixed for the round. No dense tail backward.

OPD always fused Top16→prefix draft_k, including cold. Cold corrected logits
exactly raw, but historical K-dependent ties may select different candidate IDs.
User explicitly accepted this; NO slow native K fallback is added to OPD.
Target sampling and verifier rules unchanged. Actual trajectories/round counts
may differ; no extra forwards means existing prefill/verify/expand/committed work
per round, not forcing equal totals when AAL changes.

Sparse prepares ONLY S active corrected scalars. Dense uses tiled batched rank
GEMM, reusing B tiles across contexts. Both use explicit ordered FP32 multiply/
adds (FMA/TF32 disabled) and write one reused CURRENT-proposal score workspace.
Common full-vocab scan replaces active raw values at ORIGINAL token positions,
giving bitwise same logits/topk/probability reduction across sparse/dense switches.
No probability tensor for all historical states cached. Proposal scratch reserves
O(B*C*V) for dense strategy once; sparse only touches O(S) entries. Full scan O(V),
sparse correction O(S*r); when S approaches V, measured dense strategy avoids
serial sparse loops. Canonical rank GEMM deliberately does not silently use
cuBLAS/TensorCore association if it changes exact results.

Adaptive selection reads S on GPU, never .item(). Both guarded preparation
kernels are launched; only the selected one computes. A measured threshold
profile must match GPU/Torch/Triton/CUDA/kernel hash and exact proposal geometry.
Unprofiled geometries stay sparse, no guessed nearest-shape interpolation.
Tune on B200 then verify end-to-end; RTX3090 profiles cannot select B200 default.
Counters opd_proposal_mode_sparse_rounds/dense_rounds count ROOT verification
trees, not expansion calls; cold zero-correction trees count sparse/raw.

OPD-only scheduler pads into fixed capacity, computes path lengths/EOS/prefix
extension on GPU, and transfers ONE small packet at the existing scheduling
boundary. GPU boolean padding masks/position cumulative sums replace per-round
Python padding sets/walks. Accepted real width is preserved, not replaced by
fixed-width fake EOS rows. Exact active batch size and native sampler RNG
consumption retained. One host boundary cannot be removed with DynamicCache.crop
and native dynamic batch sampling without changing those semantics.

KV: only accepted non-prefix suffix gathered into reused per-head small scratch;
write back into current storage, crop views. Accepted prefix not copied. No stack
all KV layers/full-history concat after verification. HF's next native update
creates normal contiguous KV before attention, no untested attention-layout change.
Finished batch index_select still copies remaining KV rows; further static-cache
refactor needs native-model parity and is NOT silently enabled. Historical
baseline keeps original KV work regardless of OPD improvement.

Two dependency events per rollout, side-stream update overlaps commit/KV/next
hidden work, wait only before next proposal. Profile OFF no synchronization.
Profile ON separates feature/correction/select/teacher/union/update; inclusive
proposal and overlapping wait not double-counted. End-to-end wall authoritative.

Metric definitions unchanged: exact cumulative differences per GRPO label,
AAL=sum accepted lengths / SEQUENCE verification rounds, includes target bonus,
not prefill token. KL weighted mean, other coverage/union means per valid selected
state, active rows post-update mean per OPD batch round. Nonfinite KL reported
null with counter; no gradient gate. Same CSV schema both methods.
