# OPD Reflex

## Algorithm and state

h is the actual native draft hidden state; u=h@A (FP32 projection).
A:[H,r] is a FIXED registered buffer on the EAGLE runtime adapter, initialized
deterministically once with a separate CPU generator (seed42), saved/loaded
with exported draft .pth/training resume checkpoints. Existing SpecForge
pretrain/export checkpoints without A initialize it once; broken/missing draft
weights never fall back to random mode. Rank mismatch/nonfinite A fails clearly.
A is not in an optimizer and no all-reduce is introduced during generation.

B_fast:[compactV,r] is ONE shared FP32 adapter across all active responses in
a rollout, not per response. It starts zero, changes only AFTER verification,
is reset at completion/next rollout and is never checkpointed/trained by Adam.
All proposals in a tree use the same B_t; main stream waits for update only
immediately before the next proposal (or terminal cleanup). Two dependency
events are created once per rollout when stream1; state/cache/path pools remain
owned by model cache. Only transient teacher/tree metadata gets record_stream.

Training states are the verified path plus its one-hop non-visited children,
intersected with final packed tree and feedback_contexts>=0. Never rejected
descendants recursively, never unexpanded leaves. Dedup is a GPU mask by packed
row; no Python tree walk, extra transformer call, sampled-token-only objective,
confidence gate or entropy gate.

Teacher is precisely the verification sampler's existing post-temperature/
top-p/top-k FP32 probabilities. A compact-only GPU scan computes conditional
mass/top-k; NO new full-target sort/softmax. Compact mass0/nonfinite states are
skipped with a device counter. Greedy uses the same existing argmax tokens as
a Dirac teacher, with no new softmax.

U = TopK_target_conditional UNION cached TopK_draft, defaultK16. Duplicate target
additions are masked. KL includes p_tail=1-sum_U p and q_tail=1-sum_U q; no
shortlist renormalization. Empty tail is exactly0 when U covers the whole vocab.
For coordinates j in U, the forward coarsened KL (INCLUDING tail) has gradient
q_j-p_j. One GPU batched atomic reduction applies
B[j] -= lr*sum_s(w_s*(q_sj-p_sj)*u_s)/sum_s(w_s).
No gradient/update of rows outside U. Weights sum over valid positive-weight
states across the entire batch, not separately per response. FP32 atomics can
change least significant bits; gradient tests use explicit numerical tolerance,
not a false bitwise-reproducibility claim.

## Hot path

One raw scan excludes active tokens using a tiny shared bitmap. Only active
IDs perform rank-r dots; separate sparse summaries merge with inactive top-k.
Exact top-k covers every compact token, including NEGATIVE corrections; active
raw/corrected duplicates are impossible. Proposal O(V)+O(S*r) (K fixed), with
native output-head cost unchanged. One TopK16 scan serves tree K<=8 and feedback.
No corrected/probability [all_states,V] cache, V*r backward or ANN.

Cache only normalized head inputs [B,expanded_contexts,H], u, TopK IDs/q and
normalization(max,sum). Teacher-only union tokens use selected output-head rows,
with exact SpecForge norm_output/head dtype logic. These are NOT transformer
forwards. Independent selected-dot/cuBLAS reductions can differ at a BF16 rounding
boundary; selected-head oracle tests use dtype-appropriate tolerance (1% relative
q, 1e-6 absolute for tested BF16 cases), NEVER relax target/proposal identity.

Only zero-weight B rows from exact cancellation are structurally removed, fused
into round-end index compaction. No magnitude threshold/eviction/heuristic.
Buffers reuse across compatible rollouts. Teacher workspace is bounded by TOTAL
verification_capacity+B, not B*max_verification_num. Actual S can eventually
reach V; no promise of permanently sparse state and no approximation to force it.

CUDA default is the SAME fused Top16 engine for OFF/OPD, stable lower-token-ID
tie rule and shared normalization reduction. Historical Torch softmax/topk has
different reduction/tie behavior, so upstream Torch is not asserted a bitwise
timing baseline. CPU oracle retains the legacy Torch K-dependent tree topk
(an additional debug-only topk); it is blocked on CUDA and never benchmarked as
production. CPU golden rollouts preserve original tokens/masks/history/counts.
Zero-state CUDA OFF/OPD identity is checked on the actual production kernels.

Profile segments feature/sparse-extra/state-select/teacher/union/update are
non-overlapping phases. proposal_ms is auxiliary INCLUSIVE shared proposal time;
opd_proposal_extra_ms times the added sparse correction kernel only (bitmap
scan overhead is included in inclusive wall/proposal, not hidden). Wait time may
overlap OPD-stream work: neither inclusive proposal nor wait is double-counted
in opd_profile_time_ms. End-to-end synchronized BENCHMARK wall is authoritative,
not the sum of component timers. Production adds no cuda.synchronize.

## Metrics

Each completed inherited GRPO label: one timing.csv row, same schema both modes.
step counters are EXACT cumulative differences since previous completed label.
Step AAL = delta(total_acc_length)/delta(total_decoded_token_num).
The denominator counts sequence verification rounds (not batch rounds). Accepted
length includes the target bonus emitted by verification, not the prefill token;
draft acceptance rate uses (accepted length-1)/packed proposed draft nodes.
No mean of batch ratios, cumulative substitution or moving average.

KL: weighted mean selected-state union+tail KL. Only finite weighted losses are
accumulated in opd_kl_sum; step/cumulative KL=null if that interval contains
opd_nonfinite_kl_states>0, never hides infinite loss as0. Gradient/update still
uses the prescribed finite q-p even if KL underflows; no loss-based gate.

Union size/compact mass/target mass in DraftTop16 are UNWEIGHTED means over valid
selected states. The last is conditional-compact target mass; invalid/zero
compact states are separately counted. 'Top16' field follows configured OPD_TOPK.
Active token rows: mean POST-feedback nonzero B rows per OPD verification batch,
not new rows since the last step. Verified tree nodes: mean per sequence round.
Mean active responses: sum of active responses / number of verification batches.
Verified path length equals AAL; frontier/visited is count ratio (null if no visited).
Counts/ratios are derived AFTER one small end-rollout GPU->CPU packet, never
per-round scalar transfers. OFF runs no OPD feedback/diagnostic kernels to log0.
DIAGNOSTICS1 optionally adds final B norm/max/active count to the same packet.

Forward counts can differ between POSITIVE OPD and OFF when AAL/rounds change:
zero EXTRA forwards means one target prefill+one per verification and inherited
draft prefill/expansion/committed forwards only, not artificially equal totals
when one method legitimately finishes in fewer rounds.
