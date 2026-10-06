# SpecNaacl: historical FastGRPO vs learned OPD Reflex

METHOD=fastgrpo dispatches to historical OFF c3f05ad: native FP32 softmax and
torch.topk EXACT draft_k, original Python tree/verifier/RNG/KV path. It NEVER
calls OPD's Top16 or optimized scheduler/cache. Baseline is not optimized.

METHOD=opd_reflex always uses fused Top16, including B=0/cold. It uses the
post-norm/head input, persistent LEARNED rank8 A, one shared rollout-local B.
A gradients accumulate GPU-only and apply at existing draft optimizer boundaries.
B resets each rollout. No extra target/draft transformer forward/backward per round.

Per user decision, Top16[:draft_k] ties may differ from historical K. Cold
corrected logits equal raw logits; probabilities still normalize the same full
compact distribution (FP32 reduction tolerance), no temperature/sampling change.
No slow identity fallback in OPD.

Exact sparse/dense strategy uses canonical FP32 rank GEMM and identical scan/
normalization; bitwise switch parity tested. OPD_PROPOSAL_MODE=sparse is safe
UNTUNED default. Adaptive requires measured GPU/compiler/hash/shape profile.
Historical path is never changed by OPD strategy selection.

OPD-only GPU pad masks/position IDs, one fixed scheduling packet per round and
in-place suffix KV compaction. One host boundary remains necessary for exact
HF crop/active batch/RNG behavior. No claim of zero host sync in entire decoder.

Commands: huongdanchay.md. Algorithm/metric details: METHOD_OPD_REFLEX.md.
Validation/performance caveats: IMPLEMENTATION_REPORT.md. No B200/full-model
AAL/throughput improvement claimed without real checkpoint measurements.
