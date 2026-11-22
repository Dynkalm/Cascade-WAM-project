"""Validate existing collector NPZ shards; build a disk-backed, four-way dataset.

Split unit is the source initial condition: repeated seeds of a LIBERO init are
kept together. Each branch group must be wholly contained in one input shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np

from .light_world_model import canonical_actions

FLOAT_KEYS = ("z", "future", "goal", "state", "actions", "success")
ID_KEYS = ("episode", "branch_group")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_unit(episode):
    # Existing collector IDs: suite/taskN/initN/seedN. Repeated seeds are not
    # independent initial conditions. Custom datasets should use stable IDs too.
    return re.sub(r"/seed-?\d+$", "", str(episode))


def discover(inputs):
    files = []
    for value in inputs:
        p = Path(value)
        files.extend(sorted(p.rglob("*.npz")) if p.is_dir() else [p])
    files = [p.resolve() for p in files if not p.name.endswith(".observation.npz")]
    if not files or len(set(files)) != len(files):
        raise ValueError(
            "Need unique labeled NPZ shards (observation sidecars are excluded)."
        )
    return files


def split_units(episodes, labels, seed):
    units = np.array([source_unit(e) for e in episodes])
    unique = np.unique(units)
    if len(unique) < 10:
        raise ValueError(
            "Need at least 10 independent source initial conditions for train/validation/calibration/test; this is only a technical minimum."
        )
    n = len(unique)
    nv, nc, nt = (
        max(2, round(0.15 * n)),
        max(2, round(0.10 * n)),
        max(2, round(0.15 * n)),
    )
    # Label-aware split search only; model outputs never enter the split.
    for attempt in range(1000):
        order = np.random.default_rng(seed + attempt).permutation(unique)
        chunks = dict(
            zip(
                ("validation", "calibration", "test", "train"),
                np.split(order, [nv, nv + nc, nv + nc + nt]),
            )
        )
        ids = {k: np.flatnonzero(np.isin(units, v)) for k, v in chunks.items()}
        if all(len(v) and len(np.unique(labels[v])) == 2 for v in ids.values()):
            return ids, {k: v.tolist() for k, v in chunks.items()}, attempt
    raise ValueError(
        "Cannot place both outcomes in every source-grouped split. Collect more independent successes and failures."
    )


def read_shard(path):
    with np.load(path, allow_pickle=False) as f:
        data = {k: np.asarray(f[k]) for k in FLOAT_KEYS + ID_KEYS}
        meta = json.loads(str(f["metadata"].item()))
    for k in ("feature_id", "controller", "outcome_protocol", "demo"):
        if k not in meta:
            raise ValueError(f"{path}: missing metadata {k}")
    n = len(data["success"])
    if n < 1 or any(len(v) != n for v in data.values()):
        raise ValueError(f"{path}: inconsistent row counts")
    if (
        data["z"].ndim != 3
        or data["future"].shape != data["z"].shape
        or data["actions"].ndim != 3
    ):
        raise ValueError("Expected z/future [N,P,F], actions [N,H,D]")
    if (
        data["state"].ndim != 2
        or data["goal"].ndim != 2
        or data["success"].shape != (n,)
    ):
        raise ValueError("Expected state/goal [N,D] and binary success [N]")
    for k in FLOAT_KEYS:
        if not np.isfinite(data[k]).all():
            raise ValueError(f"{path}: non-finite {k}")
    if not np.isin(data["success"], [0, 1]).all():
        raise ValueError("success must be 0 or 1")
    for k in ID_KEYS:
        if (
            data[k].shape != (n,)
            or data[k].dtype.kind not in "US"
            or np.any(data[k] == "")
        ):
            raise ValueError(f"{k} must contain nonempty string IDs [N]")
        data[k] = data[k].astype(str)
    _, valid, outside = canonical_actions(data["actions"], **meta["controller"])
    if not valid.all() or (outside > 0).any():
        raise ValueError(
            "Store exactly the canonical controller commands used for collection."
        )
    for group in np.unique(data["branch_group"]):
        ids = np.flatnonzero(data["branch_group"] == group)
        if len(np.unique(data["episode"][ids])) != 1:
            raise ValueError("A branch group crosses source episodes")
        for k in ("z", "state", "goal"):
            if not np.allclose(data[k][ids], data[k][ids[0]], atol=1e-5, rtol=1e-5):
                raise ValueError(f"{group}: sibling {k} inputs differ")
    return data, meta


def prepare(inputs, output, seed=42):
    paths = discover(inputs)
    root = Path(output)
    if root.exists():
        raise FileExistsError(
            f"{root} already exists. Reuse it for training, or choose a NEW cache path when data changes."
        )
    records, episodes, groups, labels = [], [], [], []
    metadata, shapes, seen_groups = None, None, set()
    for p in paths:
        data, meta = read_shard(p)
        spec = {k: list(data[k].shape[1:]) for k in FLOAT_KEYS}
        if metadata is None:
            metadata, shapes = meta, spec
        if spec != shapes or any(
            meta[k] != metadata[k]
            for k in ("feature_id", "controller", "outcome_protocol", "demo")
        ):
            raise ValueError(
                "Mixed feature layouts, encoders, controllers, outcome protocols, or demo/real data"
            )
        unique = set(data["branch_group"].tolist())
        if unique & seen_groups:
            raise ValueError(
                "Duplicate/split branch groups across shards; merge each group once before preparing."
            )
        seen_groups.update(unique)
        records.append(
            {"path": str(p), "sha256": digest(p), "rows": len(data["success"])}
        )
        episodes.extend(data["episode"].tolist())
        groups.extend(data["branch_group"].tolist())
        labels.extend(data["success"].tolist())
    episodes, groups, labels = np.array(episodes), np.array(groups), np.array(labels)
    splits, units, attempt = split_units(episodes, labels, seed)
    root.mkdir(parents=True)
    try:
        n = len(labels)
        arrays = {
            k: np.lib.format.open_memmap(
                root / f"{k}.npy", mode="w+", dtype=np.float32, shape=(n, *shapes[k])
            )
            for k in FLOAT_KEYS
        }
        offset = 0
        for rec in records:
            if digest(rec["path"]) != rec["sha256"]:
                raise ValueError("Input shard changed during cache creation")
            data, _ = read_shard(rec["path"])
            end = offset + rec["rows"]
            for k in FLOAT_KEYS:
                arrays[k][offset:end] = data[k]
            offset = end
        for a in arrays.values():
            a.flush()
        np.save(root / "episode.npy", episodes)
        np.save(root / "branch_group.npy", groups)
        for k, ids in splits.items():
            np.save(root / f"{k}_ids.npy", ids)
        norm = {}
        # Streaming moments avoid loading all frozen visual features into RAM.
        for key, prefix in [("state", "state"), ("actions", "action")]:
            total = square = None
            count = 0
            for start in range(0, len(splits["train"]), 1024):
                v = np.asarray(
                    arrays[key][splits["train"][start : start + 1024]], dtype=np.float64
                )
                v = v.reshape(-1, v.shape[-1])
                total = v.sum(0) if total is None else total + v.sum(0)
                square = (v * v).sum(0) if square is None else square + (v * v).sum(0)
                count += len(v)
            mean = total / count
            std = np.sqrt(np.maximum(square / count - mean * mean, 0)).clip(1e-3)
            norm[prefix + "_mean"], norm[prefix + "_std"] = mean.tolist(), std.tolist()
        # Bind the cache to the original shard bytes, split IDs, preprocessing and seed.
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "records": records,
                    "units": units,
                    "metadata": metadata,
                    "normalization": norm,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        manifest = {
            "format_version": 2,
            "fingerprint": fingerprint,
            "rows": n,
            "shapes": shapes,
            "metadata": metadata,
            "normalization": norm,
            "source_files": records,
            "split_units": units,
            "seed": seed,
            "split_search_attempt": attempt,
            "splits": {
                k: {
                    "rows": len(v),
                    "positive_rate": float(labels[v].mean()),
                    "groups": len(np.unique(groups[v])),
                }
                for k, v in splits.items()
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({"cache": str(root), **manifest["splits"]}, indent=2))
        return manifest
    except Exception:
        arrays = None
        shutil.rmtree(root)
        raise


class FeatureData:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.arrays = {
            k: np.load(self.root / f"{k}.npy", mmap_mode="r", allow_pickle=False)
            for k in FLOAT_KEYS + ID_KEYS
        }
        self.splits = {
            k: np.load(self.root / f"{k}_ids.npy") for k in self.manifest["splits"]
        }
        self.norm = {
            k: np.array(v, dtype=np.float32)
            for k, v in self.manifest["normalization"].items()
        }
        _, self.group_codes = np.unique(
            self.arrays["branch_group"], return_inverse=True
        )
        order = np.argsort(self.group_codes, kind="stable")
        cuts = np.flatnonzero(np.diff(self.group_codes[order])) + 1
        self.group_rows = {
            int(self.group_codes[ix[0]]): ix for ix in np.split(order, cuts)
        }

    def batch(self, ids):
        d = {
            k: np.array(self.arrays[k][ids], dtype=np.float32, copy=True)
            for k in FLOAT_KEYS
        }
        d["state"] = (d["state"] - self.norm["state_mean"]) / self.norm["state_std"]
        d["actions"] = (d["actions"] - self.norm["action_mean"]) / self.norm[
            "action_std"
        ]
        d["group"] = self.group_codes[ids]
        return d

    def batches(self, ids, batch_size, seed):
        # Keep complete sibling pools together so the ranking loss is meaningful.
        groups = np.unique(self.group_codes[ids])
        rng = np.random.default_rng(seed)
        allowed = np.zeros(len(self.group_codes), dtype=bool)
        allowed[ids] = True
        pending = []
        for group in rng.permutation(groups):
            row_ids = self.group_rows[int(group)]
            row_ids = row_ids[allowed[row_ids]]
            if len(row_ids) > batch_size:
                raise ValueError(
                    f"batch_size={batch_size} is smaller than a sibling pool ({len(row_ids)}); raise batch size or reduce collection K."
                )
            if len(pending) + len(row_ids) > batch_size:
                yield np.array(pending)
                pending = []
            pending.extend(row_ids.tolist())
        if pending:
            yield np.array(pending)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    prepare(a.data, a.output, a.seed)
