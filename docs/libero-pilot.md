# Matched-state LIBERO pilot

The pilot asks two questions. Can the light model preserve successful π₀.₅ candidates better than random pruning? Can the medium model use those survivors to pick a better action? The actual-future diagnostic checks whether the value heads can rank recorded outcomes when dynamics prediction error is removed.

## What the code does

| Component | Implementation |
| --- | --- |
| Candidate generation | [pi05_adapter.py](../cascade_wam/world_models/pi05_adapter.py) batches independent Gaussian noise draws through frozen OpenPI π₀.₅ and preserves its controller output transforms. The config uses 10-step chunks with seven controller coordinates. |
| Collection | [collect_libero.py](../cascade_wam/world_models/collect_libero.py) resets to an official initial state, repeats a reference action prefix, and checks the simulator state and both views plus proprioception before branching. It saves frozen features immediately after the candidate chunk or on early success. The label is task completion under a fixed, seeded π₀.₅ continuation and step budget. |
| World models | [light_world_model.py](../cascade_wam/world_models/light_world_model.py) implements both widths of the action-conditioned latent transformer. It encodes ordered actions, mixes them with the current visual tokens and proprioception, and predicts a residual update to 32 pooled visual tokens from the two cameras. The goal-conditioned head receives the endpoint features and prompt embedding. |
| Training | [prepare_training.py](../cascade_wam/world_models/prepare_training.py) groups prefixes and seeds by source initial condition and fits normalizers on training rows. [train_models.py](../cascade_wam/world_models/train_models.py) trains and saves resumable models, then calibrates the head on a separate split. |
| Evaluation and inference | [evaluate_models.py](../cascade_wam/world_models/evaluate_models.py) compares selectors and the actual-future diagnostic on held-out pools. [run_cascade.py](../cascade_wam/world_models/run_cascade.py) scores a saved pool in batches and calls the medium model only on light survivors. It exports original candidate indices. The generic [planner](../cascade_wam/planning.py) still exposes its per-candidate interface. |

| Preset | Width | Blocks | Parameters per member |
| --- | ---: | ---: | ---: |
| Light | 256 | 4 | 8,833,537 |
| Medium | 512 | 8 | 49,939,457 |

Counts assume the default π₀.₅ feature layout and exclude the frozen encoder. Both models predict one endpoint. The dynamics loss is cosine distance plus half-weighted Huber loss on latent change. After a dynamics warmup, the value head learns binary cross-entropy on actual and detached predicted futures, plus a pairwise ranking loss between successful and failed siblings. Validation cross-entropy selects the checkpoint. Calibration learns one temperature per ensemble member on the separate calibration split. No trained weights or LIBERO branch data are bundled.

## CPU smoke test

From the repo root with Python 3.9 or newer and a CPU or accelerator build of PyTorch 2.6 or newer installed:

```bash
python -m pip install -e '.[world-models]' pytest
python -m pytest -q
python -m cascade_wam.world_models.smoke_training --output /tmp/cascade-smoke-new
```

Use a fresh output directory. The smoke test trains two small versions on analytic 2-D data, evaluates held-out pools, and checks export. On the six test pools in the reference smoke run at K=8 and M=2, successful-candidate retention was **50%** versus **78.6%** in expectation for random pruning. The cascade and light-only both achieved **50%** top-1 success. This is a software check on three synthetic source initial conditions.

## Real π₀.₅ run

Use Linux, `git`, `uv`, a CUDA-capable GPU for the checkpoint, and an OpenGL backend usable by MuJoCo. The default checkpoint requires `gsutil` and network access to Google Cloud Storage. For a staged checkpoint set `CHECKPOINT` to an absolute path and ensure the upstream PaliGemma tokenizer is reachable or already in the OpenPI cache. Keep the exact checkpoint identifier and runtime for related collection and inference runs. The setup script pins OpenPI and LIBERO, installs the upstream OpenPI lockfile in Python 3.11, and adds simulator packages to that isolated environment.

The pinned runtime passed dependency and import checks. A CPU check verified native OpenPI parity using a small untrained model. A separate LIBERO check verified matched starts and shard export on one initial condition using scripted proposals. The trained-policy commands below still need a GPU run with checkpoint access.

From the repo root:

```bash
RUNTIME="$PWD/.pilot-runtime"
RUN="$PWD/outputs/libero-pilot"
bash scripts/setup_libero_pilot.sh "$RUNTIME"
PY="$RUNTIME/openpi/.venv/bin/python"
export LIBERO_CONFIG_PATH="$RUNTIME/libero-config"
export OPENPI_DATA_HOME="$RUNTIME/openpi-cache"
export MUJOCO_GL=egl
CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero

# One initial condition to verify rendering, replay, and shard creation.
"$PY" -m cascade_wam.world_models.collect_libero \
  --checkpoint "$CHECKPOINT" --suite libero_10 --tasks 0 \
  --episodes 1 --prefix-steps 0 --candidates 4 --vla-batch 1 \
  --continuation-steps 200 --output "$RUN/sanity"

OBS="$RUN/sanity/libero_10_t0_ep0_seed42_step0.observation.npz"
"$PY" -m cascade_wam.world_models.check_adapter \
  --checkpoint "$CHECKPOINT" --observation "$OBS" --candidates 4 --vla-batch 2
```

The sanity run verifies the collection path. Pool size, replay, and adapter parity are checked separately from ranking. Reduce the adapter batch to 1 if memory is limited. Run OpenPI’s official LIBERO evaluation with the same checkpoint for the closed-loop single-policy baseline. This collector uses an explicitly documented continuation policy and horizon for its offline branch labels.

Collect the training pilot in a separate directory and keep prefix, continuation and checkpoint settings fixed across comparable data:

```bash
"$PY" -m cascade_wam.world_models.collect_libero \
  --checkpoint "$CHECKPOINT" --suite libero_10 --tasks 0 1 \
  --episodes 10 --prefix-steps 0 40 80 --candidates 8 --vla-batch 1 \
  --continuation-steps 200 --seed 42 --output "$RUN/branches"

"$PY" -m cascade_wam.world_models.prepare_training "$RUN/branches" \
  --output "$RUN/cache" --seed 42

for size in light medium; do
  "$PY" -m cascade_wam.world_models.train_models --cache "$RUN/cache" \
    --size "$size" --output "$RUN/$size.pt" --device cuda
done

"$PY" -m cascade_wam.world_models.evaluate_models --cache "$RUN/cache" \
  --light-path "$RUN/light.pt" --medium-path "$RUN/medium.pt" \
  --keep 4 --device cuda --output "$RUN/report.json"
```

Use `--split validation` for development comparisons and reserve the default test split for the final comparison. Validation also selects checkpoints, so its metrics are development results. The split requires at least ten independent initial conditions and both outcomes in each split. If the pilot yields too few outcomes of either class, expand the task and initial-condition coverage before training. Choose a fresh cache path when the data changes. Resume a training run with `--resume` and preserve the corresponding `*_run` directory. The training batch must fit the largest sibling pool. For K=8, M=4 exercises pruning. A deployment claim at K=256 and M=16 needs independently labeled pools at that scale.

### Read the report

- Compare the fraction of positive pools where light pruning keeps a success against the exact random-pruning baseline. Compare cascade top-1 against light-only, medium-on-all, and random-prune-then-medium. The report includes paired differences and confidence intervals that resample source initial conditions. It also records how many pools have any successful action.
- Under `actual_future_diagnostic`, compare the same head on recorded and predicted futures. Compare the medium head on the **same predicted-light survivors** so the result isolates reranking from light-stage pruning mistakes. The recorded features are privileged offline inputs.
- Inspect `report.groups.csv` for per-pool outcomes and `report.predictions.npz` for both stages' recorded-future and predicted-future scores. Scoring-only timings exclude OpenPI sampling, encoding, model loading, and simulation.

A saved pool can be ranked without access to recorded futures:

```bash
"$PY" -m cascade_wam.world_models.run_pruner --checkpoint "$CHECKPOINT" \
  --observation "$OBS" --candidates 256 --vla-batch 1 --output "$RUN/pool.npz"
"$PY" -m cascade_wam.world_models.run_cascade --candidates "$RUN/pool.npz" \
  --light "$RUN/light.pt" --medium "$RUN/medium.pt" \
  --keep 16 --output "$RUN/selected.npz"
```

See the [improvement plan](world-model-improvements.md) for how to interpret the diagnostics and choose the next experiment.
