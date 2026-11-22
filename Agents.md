# Cascade-VLA task directory

| Open | Use it for |
| --- | --- |
| [Project.md](Project.md) | Research objective, milestones, and experimental protocol |
| [progress.md](progress.md) | Work items, their status, and evidence still needed |
| [update.md](update.md) | Dated changes, verification, and experiment references |
| [README.md](README.md) | Architecture, motivation, and research questions |
| [cascade_wam/planning.py](cascade_wam/planning.py) | Candidate, score, result, and planner interfaces |
| [tests/test_planning.py](tests/test_planning.py) | Existing planner behavior and calls to each stage |
| [pyproject.toml](pyproject.toml) | Package metadata and test discovery |

## Start here

| Order | Task | Done when |
| --- | --- | --- |
| 1 | Reproduce the frozen π₀.₅ baseline with OpenPI and LIBERO. Pin upstream revisions, document prerequisites, and start with one task and a fixed seed. | A second run can follow the recorded command and recover the same configuration and comparable outcomes. |
| 2 | Recover the candidate sampler and training artifacts from the earlier prototype. Integrate the smallest runnable pieces with reproducible commands. | The repo can export a real multi-sample candidate pool with recorded observations, seeds, action shapes, timings and checkpoint metadata. |
| 3 | Verify matched-state collection in LIBERO and implement offline oracle reporting. | The same branch can be replayed from a restored state and candidates can be compared under the same continuation policy and budget. |
| 4 | Resolve duplicate identity in the planner before relying on candidate counts and score lookups in benchmarks. Clarify the top-k API when wiring results into evaluation. | Duplicate trajectories keep distinct scores and call counts. Tests cover the documented selection or ranking behavior. |
| 5 | Train light and medium baselines, run the actual-future diagnostic, then add closed-loop comparisons. | Held-out metrics and paired runs quantify retention, reranking gains, task success, and latency. |

## Working checks

- Run `python -m pytest -q` from the repo root with pytest installed after planner changes.
- Keep model backends and simulator integration interchangeable through the planner interfaces.
- Keep every candidate pool intact across dataset splits and reuse identical pools for fixed-pool ranking comparisons. Keep oracle outcomes in the benchmark path.
- Update [progress.md](progress.md) with evidence and append implementation or experiment changes to [update.md](update.md).
