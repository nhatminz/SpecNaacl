# Fast LK Reflex

For the native EAGLE hidden feature `h`, a fixed deterministic random projection
forms a small normalized context feature:

```text
psi = normalize(h R),       dim(psi) = 8 by default
z   = z0 + A psi
```

`A psi` is the chosen fast-adapter parameterization. `R` and `A` are ordinary
non-gradient tensors. `R` is fixed for the rollout and seeded reproducibly; `A`
starts at zero separately for every response and is removed when that response
leaves the active batch. `A` is defined on SpecForge's compact EAGLE vocabulary.

Both `REFLEX_MODE=off` and `REFLEX_MODE=active` use the same EAGLE-3 compact
logits, compact softmax, top-k and fixed `d2t` mapping. The only ACTIVE-mode
difference is the additive `A psi` correction. Therefore a zero fast state is
exactly proposal-equivalent to OFF mode.

After each target verification, only the root proposal is updated. The target
token is sampled from `build_sampling_probs`, which applies the configured
temperature, top-p and top-k and returns the final normalized sampling
distribution. Reflex reuses that same tensor; it adds neither a target forward
nor a separate full-vocabulary softmax. The fixed compact-vocabulary entries
are gathered and conditioned before computing:

```text
alpha = sum_i min(p_i, q_i)
L     = -log(alpha + eps)
m_i   = 1[q_i < p_i]
S     = sum_i m_i q_i
g_i   = q_i (S - m_i) / (alpha + eps)

A <- (1 - lr * weight_decay) A - lr * g psi^T
```

Here `p` is the conditional compact-vocabulary teacher distribution:
`p_full[d2t] / (sum(p_full[d2t]) + eps)`. This is a conditional
compact-vocabulary LK objective. It is not claimed to equal the exact
full-vocabulary rejection-acceptance probability of FastGRPO's verifier.

The gradient and update are analytic batched GPU tensor operations. Reflex never
calls `backward`, creates an optimizer, changes EAGLE parameters, or adds a target
forward. The existing FastGRPO verifier decides actual acceptance exactly as
before. Persistent online EAGLE training remains a separate SpecForge operation
after rollout and is not replaced by Reflex.

The logged AAL keeps FastGRPO's weighted definition:
`total_acc_length / total_decoded_token_num` over verification rounds. Accepted
length includes the verified root/target bonus token; draft acceptance rate is
accepted draft tokens divided by proposed draft tokens.

The fast update uses an in-place batched `baddbmm`; with zero weight decay it
does not launch a separate multiply-by-one over A. Finished trajectories are
collected and compacted in one indexed operation per verification round.

The default path does not profile Reflex or log speculative rounds. Setting
`REFLEX_PROFILE=1` adds explicit synchronization around Reflex tensor work and
reports aggregate `reflex_profile_time_ms`; it is intended only for dedicated
profiling runs and does not change the algorithm.
