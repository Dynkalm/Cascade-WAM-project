"""Compare fixed selectors on held-out branch pools; not a closed-loop robot test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .light_world_model import Pruner
from .prepare_training import FeatureData, source_unit
from .run_cascade import cascade, compatible
from .train_models import binary_metrics, predict, probability, score_actual_futures


def retention_random(labels, m):
    n = len(labels)
    fail = n - int(labels.sum())
    m = min(m, n)
    return 1.0 if fail < m else 1.0 - math.comb(fail, m) / math.comb(n, m)


def random_m_heavy(labels, heavy, m):
    """Exact expectation for uniform subsets of size M and fixed stable ranking."""
    n = len(labels)
    m = min(m, n)
    order = np.argsort(-heavy, kind="stable")
    den = math.comb(n, m)
    return float(
        sum(
            labels[order[r]] * math.comb(n - r - 1, m - 1) / den
            for r in range(n - m + 1)
        )
    )


def group_table(labels, groups, light, medium, keep, *, actual_futures=None):
    rows = []
    for g in np.unique(groups):
        ids = np.flatnonzero(groups == g)
        y = labels[ids]
        n = len(ids)
        m = min(keep, n)
        li = np.argsort(-light[ids], kind="stable")
        hi = np.argsort(-medium[ids], kind="stable")
        survivors = li[:m]
        winner = survivors[np.argmax(medium[ids][survivors])]
        row = {
            "group": str(g),
            "source_unit": source_unit(str(g).rsplit("/step", 1)[0])
            if "/step" in str(g)
            else str(g),
            "k": n,
            "m": m,
            "positive_count": int(y.sum()),
            "random_top1": float(y.mean()),
            "light_top1": float(y[li[0]]),
            "medium_all_top1": float(y[hi[0]]),
            "cascade_top1": float(y[winner]),
            "random_m_medium": random_m_heavy(y, medium[ids], m),
            "oracle_available_success": float(y.max()),
            "light_retains_any_success": float(y[survivors].max()) if y.max() else None,
            "random_retains_any_success": retention_random(y, m) if y.max() else None,
        }
        if actual_futures is not None:
            light_actual, medium_actual = (
                actual_futures[name][ids] for name in ("light", "medium")
            )
            light_order = np.argsort(-light_actual, kind="stable")
            row.update(
                light_actual_top1=float(y[light_order[0]]),
                medium_actual_all_top1=float(y[np.argmax(medium_actual)]),
                medium_actual_same_survivors_top1=float(
                    y[survivors[np.argmax(medium_actual[survivors])]]
                ),
                light_actual_retains_any_success=float(y[light_order[:m]].max())
                if y.max()
                else None,
            )
        rows.append(row)
    return rows


def cluster_ci(values, units, seed=123):
    # Resample source initial conditions, not dependent branches, for a rough CI.
    unique = np.unique(units)
    rng = np.random.default_rng(seed)
    samples = []
    buckets = {u: np.flatnonzero(units == u) for u in unique}
    for _ in range(500):
        ids = np.concatenate(
            [buckets[u] for u in rng.choice(unique, len(unique), replace=True)]
        )
        samples.append(float(values[ids].mean()))
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def summarize(values, units):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "source_bootstrap_95pct_ci": cluster_ci(values, units),
    }


def paired_effect(rows, units, treatment, baseline):
    values = [row[treatment] - row[baseline] for row in rows]
    return summarize(values, units)


def evaluate(
    cache,
    light_path,
    medium_path,
    output,
    *,
    keep=4,
    batch_size=32,
    device="cpu",
    allow_demo=False,
    benchmark_groups=3,
    threads=2,
    split="test",
):
    if min(keep, batch_size, threads) < 1 or benchmark_groups < 0:
        raise ValueError(
            "keep, batch size and threads must be positive; benchmark groups cannot be negative"
        )
    torch.set_num_threads(threads)
    data = FeatureData(cache)
    if split not in {"validation", "test"}:
        raise ValueError("Choose the validation or test split")
    ids = data.splits[split]
    meta = data.manifest["metadata"]
    light, medium = [
        Pruner.load(p, device=device, allow_demo=allow_demo)
        for p in (light_path, medium_path)
    ]
    if any(
        model.dataset_fingerprint != data.manifest["fingerprint"]
        for model in (light, medium)
    ):
        raise ValueError(
            "Use both checkpoints trained on this exact cache/split manifest."
        )
    compatible(light, medium)
    if light.feature_id != meta["feature_id"]:
        raise ValueError("Feature mismatch")
    probs, actual_probs, dynamics = {}, {}, {}
    for name, pruner in [("light", light), ("medium", medium)]:
        pp = []
        observed = []
        dd = []
        for model, temp in zip(pruner.models, pruner.temperatures):
            logits, dyn = predict(model, data, ids, device, batch_size)
            pp.append(probability(logits, temp))
            actual_logits = score_actual_futures(model, data, ids, device, batch_size)
            observed.append(probability(actual_logits, temp))
            dd.append(dyn)
        probs[name] = np.mean(pp, axis=0)
        actual_probs[name] = np.mean(observed, axis=0)
        dynamics[name] = {k: float(np.mean([d[k] for d in dd])) for k in dd[0]}
    labels = np.asarray(data.arrays["success"][ids])
    groups = np.asarray(data.arrays["branch_group"][ids])
    rows = group_table(
        labels,
        groups,
        probs["light"],
        probs["medium"],
        keep,
        actual_futures=actual_probs,
    )
    units = np.array(
        [
            source_unit(
                data.arrays["episode"][ids[np.flatnonzero(groups == r["group"])[0]]]
            )
            for r in rows
        ]
    )
    for row, unit in zip(rows, units):
        row["source_unit"] = str(unit)
    names = [
        "random_top1",
        "light_top1",
        "medium_all_top1",
        "cascade_top1",
        "random_m_medium",
        "oracle_available_success",
    ]
    selection = {key: summarize([r[key] for r in rows], units) for key in names}
    eligible_mask = np.array([r["positive_count"] > 0 for r in rows])
    eligible = [r for r, valid in zip(rows, eligible_mask) if valid]
    retention = {
        k: float(np.mean([r[k] for r in eligible])) if eligible else None
        for k in ("light_retains_any_success", "random_retains_any_success")
    }
    diagnostic = {
        "scope": "Offline only. Same value heads and held-out pools with recorded instead of predicted post-action features.",
        "actual_top1": {
            key: summarize([r[key] for r in rows], units)
            for key in (
                "light_actual_top1",
                "medium_actual_all_top1",
                "medium_actual_same_survivors_top1",
            )
        },
        "light_actual_retention_given_available_success": (
            summarize(
                [r["light_actual_retains_any_success"] for r in eligible],
                units[eligible_mask],
            )
            if eligible
            else None
        ),
        "actual_minus_predicted": {
            "light_top1": paired_effect(rows, units, "light_actual_top1", "light_top1"),
            "medium_all_top1": paired_effect(
                rows, units, "medium_actual_all_top1", "medium_all_top1"
            ),
            "medium_same_survivors_top1": paired_effect(
                rows, units, "medium_actual_same_survivors_top1", "cascade_top1"
            ),
            "light_retention_given_available_success": (
                paired_effect(
                    eligible,
                    units[eligible_mask],
                    "light_actual_retains_any_success",
                    "light_retains_any_success",
                )
                if eligible
                else None
            ),
        },
    }
    warnings = []
    if split == "validation":
        warnings.append(
            "Validation was used for checkpoint selection; treat these metrics as development results."
        )
    if all(r["m"] == r["k"] for r in rows):
        warnings.append(
            "M >= K in every test group: there is NO pruning; retention is trivial. Lower M or collect larger pools."
        )
    if len(np.unique(units)) < 20:
        warnings.append(
            "Few independent held-out initial conditions; confidence intervals and success estimates are unstable."
        )
    if not any(0 < r["positive_count"] < r["k"] for r in rows):
        warnings.append(
            "No test pool mixes successes and failures; selection quality cannot be distinguished."
        )
    timings = []
    # Actual calls, using pre-encoded observations; no cached heavy-all scores in
    # cascade timing. Warm-up first. Excludes VLA, simulation, loading and encoding.
    for group in np.unique(groups)[:benchmark_groups]:
        ix = ids[groups == group]
        features = {k: np.array(data.arrays[k][ix[0]]) for k in ("z", "state", "goal")}
        actions = np.asarray(data.arrays["actions"][ix])
        m = min(keep, len(ix))
        funcs = {
            "light_all": partial(
                light.rank,
                features,
                actions,
                feature_id=meta["feature_id"],
                keep=1,
                batch_size=batch_size,
            ),
            "medium_all": partial(
                medium.rank,
                features,
                actions,
                feature_id=meta["feature_id"],
                keep=1,
                batch_size=batch_size,
            ),
            "cascade": partial(
                cascade,
                light,
                medium,
                features,
                actions,
                meta["feature_id"],
                m,
                1,
                batch_size,
            ),
        }
        timing = {"group": str(group), "k": len(actions), "m": m}
        for name, fn in funcs.items():
            fn()
            measure = []
            for _ in range(3):
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t = time.perf_counter()
                fn()
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                measure.append(time.perf_counter() - t)
            timing[name + "_seconds"] = float(np.median(measure))
        timings.append(timing)
    train_rate = float(data.arrays["success"][data.splits["train"]].mean())
    report = {
        "scope": "Offline selection among fixed, observed branch outcomes. NOT closed-loop policy success or end-to-end robot latency.",
        "demo": meta["demo"],
        "device": device,
        "split": split,
        "evaluated_rows": len(ids),
        "evaluated_groups": len(rows),
        "source_units": len(np.unique(units)),
        "requested_keep": keep,
        "selection": selection,
        "paired_selection_effects": {
            "cascade_minus_light_top1": paired_effect(
                rows, units, "cascade_top1", "light_top1"
            ),
            "cascade_minus_medium_all_top1": paired_effect(
                rows, units, "cascade_top1", "medium_all_top1"
            ),
            "cascade_minus_random_prune_medium_top1": paired_effect(
                rows, units, "cascade_top1", "random_m_medium"
            ),
        },
        "retention_conditioned_on_available_success": {
            "eligible_groups": len(eligible),
            **retention,
            "light_minus_random": (
                paired_effect(
                    eligible,
                    units[eligible_mask],
                    "light_retains_any_success",
                    "random_retains_any_success",
                )
                if eligible
                else None
            ),
        },
        "classification": {k: binary_metrics(labels, v) for k, v in probs.items()},
        "train_rate_constant_baseline": binary_metrics(
            labels, np.full(len(labels), train_rate)
        ),
        "future_prediction": dynamics,
        "actual_future_diagnostic": diagnostic,
        "scoring_only_timings": timings,
        "timing_excludes": [
            "VLA sampling",
            "feature encoding",
            "checkpoint loading",
            "simulation",
        ],
        "warnings": warnings,
        "dataset_fingerprint": data.manifest["fingerprint"],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    with open(output.with_suffix(".groups.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output.with_suffix(".predictions.npz"),
        row_ids=ids,
        success=labels,
        branch_group=groups,
        light=probs["light"],
        medium=probs["medium"],
        light_actual=actual_probs["light"],
        medium_actual=actual_probs["medium"],
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--light-path", required=True)
    p.add_argument("--medium-path", required=True)
    p.add_argument("--output", default="comparison.json")
    p.add_argument("--keep", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-demo", action="store_true")
    p.add_argument("--benchmark-groups", type=int, default=3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate(**vars(p.parse_args()))
