# Research progress

**Pilot integration** · 2026-11-23

The Python planner, batched world models, and offline LIBERO pilot are integrated. The CPU workflow is verified with synthetic data. Real π₀.₅ checkpoint inference, GPU model training, and held-out LIBERO ranking remain open.

| Needed change | Status | Evidence and remaining work |
| --- | --- | --- |
| Project plan and handoff | Complete | [Project plan](Project.md), [change log](update.md), [task directory](Agents.md), and the [pilot guide](docs/libero-pilot.md) describe the workflow. |
| Two-stage planner | Implemented and unit-tested | The generic planner handles per-candidate scorers. The batched inference path preserves candidate indices and calls medium only on the survivors. |
| Candidate identity and ranking API | Open in the generic planner | Duplicate trajectory keys can overwrite its score entries. The batched path tracks original indices. Clarify the generic `top_k` behavior when integrating a closed-loop consumer. |
| Reproducible OpenPI/LIBERO baseline | Pinned runtime verified | Dependency and import checks passed. The adapter matched native batched outputs with a small untrained OpenPI model on CPU. Run the official single-policy baseline and matched-noise parity with the trained checkpoint. |
| Multi-sample export | Implemented and unit-tested with a stub | Small unranked exports no longer require a survivor limit. Verify diversity, clipping, latency, and peak memory on actual action samples. |
| Matched-state collection and oracle | Simulator replay verified with scripted proposals | The collector rendered LIBERO, verified matched starts, and saved readable branch shards. The evaluator calculates offline oracle availability. Collect mixed-outcome pools with the trained policy next. |
| Light and medium training | Implemented and smoke-tested on CPU | Both models train, calibrate and export on synthetic data. Train and compare on actual LIBERO branch outcomes. |
| Actual-future diagnostic | Implemented and smoke-tested on CPU | Uses the same heads on actual and predicted endpoints. Medium reranking is compared on exactly the same light survivors. Run the comparison on real held-out pools. |
| K/M sweeps and closed-loop evaluation | Planned | Use the [improvement plan](docs/world-model-improvements.md) to select ablations. Measure paired success, latency, and memory on actual rollouts. |

Update each row when its evidence changes. Link the relevant commit, test output, experiment manifest, or result table, and record reproducible commands for new runs in [update.md](update.md).
