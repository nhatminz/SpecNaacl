# SpecNaacl — FastGRPO vs OPD Reflex

Current methods only: `METHOD=fastgrpo` and `METHOD=opd_reflex`.
LK Reflex has been retired. Per-model ordinary launchers select OPD; suffix
`_fastgrpo.sh` selects the fair in-repository baseline.

Both methods share the same EAGLE-3, compact vocabulary, fused Top16 proposal
engine, tree construction, target sampler/verifier, GRPO loss/rewards, real
SpecForge online draft loss/update schedule, checkpoint and step telemetry.
OPD alone adds a fixed checkpoint-persistent rank-8 projector and one shared
rollout-local fast adapter, learning from visited/expanded one-hop frontier
states with existing verification probabilities. No extra transformer forward.

Default dataset is simplelr, target/draft LR1e-5, batch8, accumulation4,
responses8, draft microbatch token budget2048, log1. OPD LR0.01 is configurable,
TopK16/rank8, profiling/diagnostics OFF. Stream0 is UNTUNED conservative default:
benchmark0/1 on your B200 before claiming which is faster.

Important exactness boundary: CUDA's shared fused engine uses a deterministic
low-token-ID tie rule and its FP32 reduction. It is NOT asserted bitwise identical
to historical Torch topk/softmax on ties. OFF and zero-state OPD in THIS same
backend are bitwise tested including tree/verifier/tokens/RNG/counts. CPU oracle
retains legacy Torch's K-dependent tie rule (a debug-only additional topk);
it is not a production timing baseline.

Read [huongdanchay.md](huongdanchay.md), [METHOD_OPD_REFLEX.md](METHOD_OPD_REFLEX.md)
and [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md).
No model/data/checkpoint paths or dependency pins have been replaced. The
existing SpecForge architecture, feature capture and training-time unrolling are
unchanged (vendored commit3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83).
No pretrained assets/B200 measurements are invented.
