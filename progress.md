# Research progress

**Baseline reviewed** · 2026-11-23 · repository revision `1acf741`

The repository currently contains the Python selection harness and three passing unit tests. Separate conversation artifacts were reported to contain candidate collection and model training code exercised on synthetic data. Real π₀.₅ inference, LIBERO collection, and GPU model training remain unverified in this repo.

| Needed change | Status | Evidence and remaining work |
| --- | --- | --- |
| Project plan and handoff | Complete | [Project plan](Project.md), this tracker, [change log](update.md), and [task directory](Agents.md) are written. |
| Two-stage planner | Implemented and unit-tested | Each proposal receives a light score, feasible candidates are capped before medium scoring, and an empty survivor set returns no selection. Existing tests passed on Python 3.12. |
| Candidate identity and ranking API | Open | Duplicate trajectory keys can overwrite score entries. Establish per-proposal identity and cover identical trajectories in a regression test. Decide and document whether to expose ranked top-k results or retain the current single-selection API. |
| Reproducible OpenPI/LIBERO baseline | Open | Pin the upstream revisions, record the runtime and hardware requirements, and run the official `pi05_libero` baseline on a small documented task and seed set. |
| Multi-sample candidate export | Prototype reported externally | Bring the π₀.₅ sampler and exporter into the repo. Verify independent sampling noise and action diversity on actual observations, then measure memory and latency at increasing batch sizes. |
| Matched-state branch collection and oracle | Prototype reported externally | Bring in collection code, verify exact restore and reproducible outcomes on LIBERO, then collect and audit a first labeled pool dataset. |
| Lightweight training and integration | Prototype reported externally | Import and test the approximately 8.8M-parameter training workflow. Train on real branch data, assess survivor retention on held-out pools, and wire inference into the planner. |
| Medium training and representation diagnostic | Prototype reported externally | Import the approximately 49.9M-parameter training workflow. Compare actual-future and predicted-future scoring on shared held-out pools, then measure the medium model's gain on identical survivor sets. |
| Closed-loop cascade and ablations | Open | Compare the [planned baselines](Project.md#evaluation-protocol) on paired seeds. Measure success, latency and peak memory. Use the results to choose K, M, and any verifier upgrade. |

Update each row when its evidence changes. Link the relevant commit, test output, experiment manifest, or result table, and record reproducible commands for new runs in [update.md](update.md).
