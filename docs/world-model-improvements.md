# World-model improvement plan

The highest-value decision is whether the models fail to **predict** the useful consequences of an action or fail to **recognize** them from the representation they receive. More parameters address only some of these cases. Use a held-out development split for design decisions and keep a final test split untouched.

| Observation on matched pools | Next experiment | How to judge it |
| --- | --- | --- |
| Both heads struggle with recorded futures. | Compare a value probe on frozen tokens with probes on finer wrist-camera patches, robot state, or object-relative motion. Use simulator object state as an offline diagnostic to check whether the label is separable at all. | Recorded-future top-1 and retention improve on mixed-outcome pools. Add a current-observation-only baseline to expose task or starting-state shortcuts. |
| Recorded futures rank well, but predicted futures rank poorly. | Predict two or three intermediate frames, with heavier loss near the gripper and moving objects. Keep the current endpoint predictor as the control. Start on one task with grasp failures. | Predicted-vs-recorded top-1 and retention gaps shrink, especially on near-identical grasp and miss pairs. Compare against copying the current features and report gains at equal inference cost. |
| Light pruning loses successful branches. | Train a direct action-conditioned value scorer as a cheap baseline. Compare it with the latent predictor under a within-pool objective that emphasizes the top of the list. Consider a larger M for ambiguous pools only after measuring disagreement calibration. | Compare success retention at fixed M and fixed latency against random M and the existing light model. Report how often an ambiguity rule increases compute. |
| Light pruning retains winners but medium reranking adds little. | Compare the current medium stage with a verifier using richer wrist and intermediate features. Include a wider endpoint predictor at comparable cost to distinguish richer inputs from parameter count. | Measure paired top-1 improvement on identical light survivor sets. Also report losses from reranking actions the light stage got right. |
| Candidate pools rarely contain a winner, or proposals repeat. | Measure pairwise chunk distance, duplicates after clipping, and oracle availability as K increases. Compare independent-noise sampling with cheap deduplication and fewer distinct proposals. | Plot oracle availability against sampling latency and effective pool size. A flat oracle curve limits any selector improvement. |
| Reranking helps offline but end-to-end latency stays high. | Profile VLA sampling, feature encoding, and both scorers separately. Benchmark shared observation work across candidate batches after native parity is verified. Compare fixed and adaptive survivor counts. | Plot paired closed-loop success against p50/p95 decision time and peak GPU memory. Use the full pipeline when reporting latency. |

## Compute comparison

With per-candidate costs `c_light` and `c_medium`, the cascade saves scorer work when `K*c_light + M*c_medium < K*c_medium`. This is a starting estimate. Actual batch throughput, memory and device synchronization determine wall-clock savings. The VLA proposes all K candidates either way, so measure its share of total latency before investing in a faster reranker.

## Experiment sequence

1. **Validate the pilot on real LIBERO data.** Verify replay, native OpenPI parity, label balance, and oracle coverage. Use the existing endpoint models to measure the two diagnostic gaps before changing architecture.
2. **Run the smallest ablation that addresses the observed failure.** Use the same pools, split, continuation budget, and seeds. Report paired effects, source-level uncertainty, and subgroup counts. Treat the first small pilot as directional.
3. **Scale only the promising variants.** Sweep K and M, including larger labeled pools, and compare with single-sample π₀.₅, light-only, all-medium, and random pruning. Evaluate unseen tasks separately from held-out initial conditions of known tasks.
4. **Verify in closed loop.** Replan after a fixed chunk or prefix, measure real decision latency, and use paired task/seed schedules. A cascade is worth keeping if it improves success over light-only and offers a better success–latency tradeoff than all-medium.

On the synthetic smoke run, substituting recorded futures changed neither top-1 success nor light retention. Both light-only and the cascade achieved 50% top-1 success, and all-medium achieved 66.7%. These six toy pools show how to read the report. Model size, simulator performance, and the value of the second stage remain open. The recommended first comparison is recorded-future versus predicted-future rankings because it uses the same trained heads and can be run before collecting intermediate targets.
