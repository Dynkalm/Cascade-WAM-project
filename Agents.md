# Cascade-VLA task directory

| Open | Use it for |
| --- | --- |
| [Project.md](Project.md) | Research objective, milestones, and experimental protocol |
| [progress.md](progress.md) | Work items, their status, and evidence still needed |
| [update.md](update.md) | Dated changes, verification, and experiment references |
| [README.md](README.md) | Architecture, motivation, and research questions |
| [cascade_wam/planning.py](cascade_wam/planning.py) | Candidate, score, result, and planner interfaces |
| [tests/test_planning.py](tests/test_planning.py) | Existing planner behavior and calls to each stage |
| [Pilot guide](docs/libero-pilot.md) | Runtime setup, collection, training, and report commands |
| [Improvement plan](docs/world-model-improvements.md) | Ablations and the evidence that would justify each change |
| [pyproject.toml](pyproject.toml) | Package metadata and test discovery |

## Start here

| Order | Task | Done when |
| --- | --- | --- |
| 1 | Run the [real π₀.₅ sanity check](docs/libero-pilot.md#real-π₀₅-run) on a GPU with checkpoint access. | One task produces a replay-verified shard and the batched adapter matches native inference for the same noise. |
| 2 | Run the official frozen π₀.₅ LIBERO baseline and collect the matched-state pilot. | Record task IDs, seeds, checkpoint, and timing. The pilot has enough initial conditions and mixed outcomes for grouped evaluation. |
| 3 | Train both endpoint models and run the actual-future diagnostic. | Report light retention against random pruning and medium gains on identical survivors, with paired effects and source-level uncertainty. |
| 4 | Choose an ablation using the [improvement plan](docs/world-model-improvements.md). | A held-out development comparison isolates the likely bottleneck and justifies scaling to larger labeled pools. |
| 5 | Add closed-loop cascade comparisons and profile the complete decision path. | Paired task/seed runs measure success, latency, and peak memory against single-sample π₀.₅, light-only, and all-medium. |

## Working checks

- Run `python -m pytest -q` from the repo root with pytest installed after planner changes.
- Keep model backends and simulator integration interchangeable through the planner interfaces.
- Keep every candidate pool intact across dataset splits and reuse identical pools for fixed-pool ranking comparisons. Keep oracle outcomes in the benchmark path.
- Update [progress.md](progress.md) with evidence and append implementation or experiment changes to [update.md](update.md).
