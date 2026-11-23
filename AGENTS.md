# Cascade-VLA repository guide

## Changelog rule

Every change to the repository must include an entry in [update.md](update.md) in the same change. This applies to code, configuration, documentation, tests, and recorded experiments. Use the newest date section and the applicable headings from the [changelog convention](update.md#entry-format). Summarize what changed, record the checks run and their outcome when relevant, and link to experiment evidence when available.

When a change affects the system or a model, update the matching section in [Project.md](Project.md). Record where the design came from and whether it is implemented, an experimental control, or still proposed. Update [progress.md](progress.md) as evidence and open tasks change.

## Repository structure

```text
.
├── README.md                        Overview and original research proposal
├── Project.md                       Research goals, model architectures, and system design
├── progress.md                      Status, open tasks, and supporting evidence
├── update.md                        Dated project changelog
├── AGENTS.md                        Repository map and contributor instructions
├── pyproject.toml                   Packaging, optional PyTorch/NumPy, and test discovery
├── .gitignore                       Generated runtime, outputs, and cache exclusions
├── cascade_wam/
│   ├── __init__.py                  Public generic planner API
│   ├── planning.py                  Pluggable proposal generation and two-stage selection
│   └── world_models/
│       ├── __init__.py              Pilot package
│       ├── pi05_adapter.py          Frozen OpenPI sampling and feature extraction
│       ├── check_adapter.py         Native OpenPI parity with matching action noise
│       ├── collect_libero.py        Matched-state simulation branches and outcome shards
│       ├── prepare_training.py     Shard validation, grouped splits, and feature cache
│       ├── light_world_model.py     Shared latent predictor and scorer for both sizes
│       ├── train_models.py          Presets, objectives, checkpoints, and calibration
│       ├── evaluate_models.py       Held-out ranking, actual-future diagnostic, and reports
│       ├── run_pruner.py            π₀.₅ pool export and optional light-only scoring
│       ├── run_cascade.py           Batched light pruning and medium reranking
│       └── smoke_training.py       Synthetic CPU workflow for training and evaluation
├── docs/
│   ├── libero-pilot.md              Environment setup and experiment commands
│   └── world-model-improvements.md  Architecture comparisons and scaling experiments
├── scripts/
│   └── setup_libero_pilot.sh        Pinned OpenPI and LIBERO runtime setup
└── tests/
    ├── test_planning.py             Generic planner behavior and model-call counts
    └── test_world_models.py         Replay checks, training, diagnostics, and export tests
```

The package exports the generic planner from [cascade_wam/](cascade_wam/). The experimental commands live in [world_models/](cascade_wam/world_models/) and run from the repo root with `python -m cascade_wam.world_models.<module>`. The [pilot guide](docs/libero-pilot.md) lists the specific flags and output formats. Local runtime dependencies and experiment artifacts go under the ignored `.pilot-runtime/` and `outputs/` directories in the guide's examples.

## Start here

| Order | Task | Done when |
| --- | --- | --- |
| 1 | Run the [real π₀.₅ sanity check](docs/libero-pilot.md#real-π₀₅-run) on a GPU with checkpoint access. | One task produces a replay-verified shard and the batched adapter matches native inference for the same noise. |
| 2 | Run the official frozen π₀.₅ LIBERO baseline and collect the matched-state pilot. | Record task IDs, seeds, checkpoint, and timing. The pilot has enough initial conditions and mixed outcomes for grouped evaluation. |
| 3 | Train both endpoint models and run the actual-future diagnostic. | Report light retention against random pruning and medium gains on identical survivors, with paired effects and source-level uncertainty. |
| 4 | Choose an ablation using the [improvement plan](docs/world-model-improvements.md). | A held-out development comparison isolates the likely bottleneck and justifies scaling to larger labeled pools. |
| 5 | Add closed-loop cascade comparisons and profile the complete decision path. | Paired task and seed runs measure success, latency, and peak memory against single-sample π₀.₅, light-only, and all-medium. |

## Working checks

- Run `python -m pytest -q` from the repository root after code changes. Install the optional `world-models` extra and pytest to exercise the pilot tests.
- Keep model backends and simulator integration interchangeable. Reuse the relevant planner or batched pilot interfaces when extending them.
- Keep every candidate pool intact across dataset splits and reuse identical pools for fixed-pool ranking comparisons. Keep oracle outcomes in the benchmark path.
