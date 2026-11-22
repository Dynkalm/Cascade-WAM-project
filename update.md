# Update log

## 2026-11-23

- Integrated batched candidate generation, matched-state collection, light and medium training, and offline inference. Added recorded-versus-predicted future scoring and paired selection effects.
- Added pinned runtime setup, pilot commands, and a failure-driven model improvement plan. Fixed the small unranked export argument check.
- Verified all 15 repo tests and both stages of the synthetic CPU smoke test. The synthetic cascade matched light-only top-1 selection and pruned successful candidates more often than the random baseline.
- Verified installation of the pinned runtime, native OpenPI parity with a small untrained CPU model, and branch replay in LIBERO with scripted proposals. Trained π₀.₅ inference and LIBERO ranking results remain pending.

### Earlier on 2026-11-23

- Added the research plan, progress tracker, and task directory. The roadmap prioritizes the frozen π₀.₅/LIBERO baseline and a matched-state oracle before using both learned world models in closed loop.
- Reviewed GitHub `main` at `1acf741`. It includes the initial two-stage planner from `3f25a53` and the revised research README.
- Verified the planner test suite with `python -m pytest -q` on Python 3.12 with pytest 9.2.0. All three tests passed, covering survivor pruning, reranking, and an empty survivor set.
- Recorded the candidate collector and light/medium training prototypes described in conversation as external work. Their reported synthetic checks are not repo test results.

For future entries, add the date, change, and verification command or result link. Record the checkpoint, dataset or artifact reference, and run configuration when logging experiments.
