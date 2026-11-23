# Cascade-VLA changelog

This is the chronological record of repository changes and experiment results. Keep newest dates first.

## Entry format

For every repository change, add a bullet here in the same commit or pull request. Use a `## YYYY-MM-DD` date heading and the applicable categories below. Reuse an existing category under that date.

| Heading | What to record |
| --- | --- |
| Added | New capabilities, tests, or documentation |
| Changed | Updates to existing behavior, model choices, configuration, or docs |
| Fixed | Defects corrected and the affected behavior |
| Removed | Capabilities, dependencies, or documentation removed |
| Verified | Commands and observed results, with any relevant limits |

Start each change bullet with a verb and include enough detail to locate the affected component. For experiments, record the configuration, checkpoint, dataset or artifact reference, and verification command or result link when available. Identify work in progress as such. Empty categories can be omitted.

## 2026-11-23

### Changed

- Expanded [Project.md](Project.md) into the architecture reference for frozen π₀.₅, both world-model presets, the batched cascade, the offline learning path, and the planned closed loop. Documented which choices came from the uploaded prototype and which design questions remain open.
- Renamed the repository guide to [AGENTS.md](AGENTS.md), added a directory map and a standing changelog rule, and updated the guide links in the project and progress documents.
- Converted this log to dated changelog categories and retained the earlier change and verification history.

### Added

- Documented a scaling comparison for 256, 1,024, and 4,096 proposals and larger endpoint predictors. [Scaling plan](docs/world-model-improvements.md#candidate-pool-scale)
- Integrated batched candidate generation, matched-state collection, light and medium training, and offline inference. Added recorded-versus-predicted future scoring and paired selection effects. [Pilot guide](docs/libero-pilot.md)
- Added a pinned runtime setup, pilot commands, and a model improvement plan organized around observed failures. [Improvement plan](docs/world-model-improvements.md)
- Added the initial research plan, progress tracker, and task directory. The roadmap prioritized a frozen π₀.₅/LIBERO baseline and a matched-state oracle before using both learned world models in closed loop.
- Recorded the externally developed collector and light and medium prototypes and the synthetic checks reported with them.

### Fixed

- Allowed small unranked exports without applying the ranked survivor limit.

### Verified

- Checked all 54 relative documentation links and confirmed that the repository map covers all 25 tracked paths. Ran `python -m pytest -q` and passed all 15 tests.
- Ran CPU forward checks on the 23.6M, 151.6M, and 266.7M configurations using synthetic inputs. A stub sampler verified 4,096-candidate batching and a nested 256-candidate prefix. Trained-policy diversity, latency, and ranking at these sizes remain unmeasured.
- Passed 15 repository tests and trained both stages in the synthetic CPU workflow. On the toy pools the cascade matched light-only top-1 success. The light stage kept at least one successful candidate in 50% of the pools, compared with an expected 78.6% for random pruning.
- Verified the pinned runtime installation, native OpenPI parity with a small untrained CPU model, and LIBERO branch replay with scripted proposals. Trained π₀.₅ inference and LIBERO ranking results remain pending.
- Reviewed GitHub `main` at `1acf741`, which included the initial two-stage planner from `3f25a53` and the revised research README. Ran `python -m pytest -q` on Python 3.12 with pytest 9.2.0. Three planner tests passed, covering survivor pruning, reranking, and an empty survivor set.
