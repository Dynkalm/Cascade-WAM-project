# Cascade-VLA project plan

Cascade-VLA studies whether a lightweight, action-conditioned world model can filter action chunks proposed by a frozen π₀.₅ policy while preserving enough good options for a stronger world model to make the final choice. The goal is to match or improve LIBERO task success at lower end-to-end decision cost than scoring every proposal with the stronger model.

## System and experimental scope

1. Use the official OpenPI `pi05_libero` checkpoint. Establish the standard single-sample policy baseline, then sample multiple action chunks from the same observation and instruction using independent initial noise. Record seeds and sampling settings.
2. Score all **K** proposals with a lightweight predictor using the current observation, instruction, and proposed action chunk. Retain **M** candidates with promising goal progress or success estimates and track feasibility and uncertainty where measured.
3. Rerank those **M** candidates with the medium model and execute the selected chunk or a fixed prefix. Obtain a new observation and replan. Keep the execution horizon and replanning policy fixed across methods.
4. In the benchmark only, restore the same simulator state for every alternative. Execute the candidate, then use a fixed π₀.₅ continuation policy with a documented budget and seed protocol. Label each branch with actual task success under that protocol. Use these outcomes for training, evaluation, and an offline oracle upper bound.

The control uses **K=256**, **M=16**, and a 10-step action chunk. The lightweight and medium controls use approximately **8.8M** and **49.9M** trainable parameters, respectively, excluding the frozen encoder. Both predict an endpoint in frozen feature space and learn task-success and within-pool ranking signals. The [scaling comparisons](docs/world-model-improvements.md#candidate-pool-scale) extend the proposed K range to **4,096** and test larger rerankers up to **266.7M** parameters.

## Milestones and exit checks

| Milestone | Deliverable | Exit check |
| --- | --- | --- |
| 1. π₀.₅ baseline | Reproducible OpenPI and LIBERO commands, run metadata, and candidate pool export | Run the standard policy on a documented suite and seeds. Verify distinct noise draws, valid outputs, and report action diversity plus sampling latency. |
| 2. Matched-state benchmark | Branch collector, outcome labels, and offline oracle | Verify restoration of the simulator state and repeatability of a fixed branch. Report pool sizes, outcome mix, the share of pools with a successful candidate, and the resulting oracle gap. |
| 3. Lightweight pruner | Trained endpoint predictor and goal-conditioned ranker | Evaluate on held-out pools with successful candidates. Report how often at least one success survives, Recall@M against the chosen oracle ranking, and ranking and calibration quality. |
| 4. Medium reranker | Trained medium model, planner adapters, and representation diagnostic | Score actual post-action features and predicted features on the same held-out pools. Measure whether reranking improves final selection over the light model on the same survivors. |
| 5. Closed-loop evaluation | Reproducible comparison with cost and task outcomes | Compare paired runs with uncertainty estimates. Report task success, decision latency by stage, peak GPU memory, and compute–success curves across K and M. |

## Evaluation protocol

- Reuse the same saved action pools and simulator starts for fixed-pool ranking comparisons. Use paired task and seed schedules for closed-loop comparisons, where each policy's actions can change its subsequent observations.
- Compare single-sample π₀.₅, random selection from K, light-only, medium on all K, random M followed by medium scoring, the cascade, and the offline simulator oracle.
- Keep all alternatives from a starting state in the same data split. Record suite, task, episode or reset state, policy checkpoint, action horizon, continuation budget, seeds, and feature configuration. Use separate training, validation, calibration, and test splits.
- Report success retention conditional on a successful proposal existing, alongside candidate-pool coverage and final success. Measure actual closed-loop batch latency as well as evaluation counts.
- Use the true-future diagnostic to prioritize upgrades. If actual future features give weak rankings, investigate pooled features, labels, and the scoring head. If true futures rank well but predicted futures do poorly, test intermediate predictions or more spatial and object-level features before committing to a substantially bigger verifier.

To reproduce the first experiment, follow the [LIBERO pilot guide](docs/libero-pilot.md). The [model improvement plan](docs/world-model-improvements.md) maps observed failure modes to experiments. For live status and the next executable tasks, see [progress.md](progress.md) and [Agents.md](Agents.md). See [README.md](README.md) for the original architecture and research questions.
