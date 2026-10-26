# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare isolated LBench RECAP datasets using the reference LeRobot statistics.

Run with the local LeRobot v3 environment, whose histogram implementation
reproduces the supplied reference exactly. Training uses the v2.1 environment.
Source datasets and the initial policy checkpoint are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats

FEATURES = {"observation.state": "state", "action": "actions"}
STAT_KEYS = ("mean", "std", "q01", "q99")
DATA_ROOT = Path("/root/autodl-tmp/datasets")
REFERENCE = (
    DATA_ROOT
    / "lbench_8tasks_all_merge_simple_v21_merged"
    / "assets/lerobot_meta_stats/norm_stats.json"
)
SOURCES = {
    "sft": DATA_ROOT / "lbench_8tasks_all_merge_simple_v21",
    "rollout": DATA_ROOT / "round1_success_failure",
}


def read_jsonl(path: Path) -> list[dict]:
    """Read nonempty JSONL records."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def success_labels(path: Path, episode_ids: set[int]) -> dict[int, bool]:
    """Require one explicit binary outcome for every rollout episode."""
    labels = {}
    for row in read_jsonl(path):
        ep = int(row["episode_index"])
        value = row.get("source_episode_success")
        if ep in labels or value not in (0, 1, False, True):
            raise ValueError(f"Duplicate or invalid episode outcome: {path}, {ep}")
        if row.get("source_episode_done") is not True:
            raise ValueError(f"Incomplete rollout: {path}, {ep}")
        labels[ep] = bool(value)
    if set(labels) != episode_ids:
        raise ValueError(f"Rollout outcomes do not cover episode IDs: {path}")
    return labels


def compute_episode(path: Path) -> dict:
    """Compute raw-frame statistics separately for each feature dimension."""
    table = pq.read_table(path, columns=[*FEATURES, "episode_index", "frame_index"])
    episodes = table["episode_index"].to_numpy()
    if not len(episodes) or len(np.unique(episodes)) != 1:
        raise ValueError(f"Expected exactly one nonempty episode: {path}")
    np.testing.assert_array_equal(
        table["frame_index"].to_numpy(), np.arange(len(episodes))
    )
    result = {}
    for key in FEATURES:
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)
        if values.shape != (len(episodes), 8) or not np.isfinite(values).all():
            raise ValueError(f"Invalid LBench feature: {path}, {key}")
        # The reference uses 5000-bin LeRobot histograms within each episode.
        # np.quantile over pooled frames or action windows is not equivalent.
        result[key] = get_feature_stats(values, axis=0, keepdims=False)
    return result


def openpi_stats(stats: dict) -> dict:
    """Map names and cast once after episode aggregation, like the reference."""
    return {
        "norm_stats": {
            name: {
                s: np.asarray(stats[key][s], dtype=np.float32).tolist()
                for s in STAT_KEYS
            }
            for key, name in FEATURES.items()
        }
    }


def write_json(path: Path, value: dict) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare_dataset(source: Path, destination: Path, kind: str) -> dict:
    """Copy metadata; link videos; add verified outcomes to rollout parquet copies."""
    info = json.loads((source / "meta/info.json").read_text())
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    ids = {int(row["episode_index"]) for row in episodes}
    if len(ids) != len(episodes) or len(ids) != info["total_episodes"]:
        raise ValueError(f"Episode metadata mismatch: {source}")
    labels = (
        success_labels(source / "meta/rollout_provenance.jsonl", ids)
        if kind == "rollout"
        else dict.fromkeys(ids, True)
    )
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "meta").exists():
        shutil.copytree(source / "meta", destination / "meta")
    if not (destination / "videos").exists():
        (destination / "videos").symlink_to(source / "videos", target_is_directory=True)
    if kind == "sft":
        if not (destination / "data").exists():
            (destination / "data").symlink_to(source / "data", target_is_directory=True)
    else:
        for file in sorted((source / "data").rglob("*.parquet")):
            target = destination / file.relative_to(source)
            table = pq.read_table(file)
            outcomes = [labels[int(ep)] for ep in table["episode_index"].to_numpy()]
            table = table.append_column(
                "is_success", pa.array(outcomes, type=pa.bool_())
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
        info["features"]["is_success"] = {"dtype": "bool", "shape": [1], "names": None}
        write_json(destination / "meta/info.json", info)
    for ep in episodes:
        for camera in (k for k, f in info["features"].items() if f["dtype"] == "video"):
            video = info["video_path"].format(
                episode_chunk=ep["episode_index"] // info["chunks_size"],
                video_key=camera,
                episode_index=ep["episode_index"],
            )
            if not (destination / video).is_file():
                raise FileNotFoundError(destination / video)
    return {
        "source": str(source),
        "dataset_path": str(destination),
        "type": kind,
        "episodes": len(episodes),
        "frames": sum(ep["length"] for ep in episodes),
        "successes": sum(labels.values()),
        "failures": len(labels) - sum(labels.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DATA_ROOT / "lbench_recap_round1"
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest_path = args.output / "preparation_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Already prepared: {manifest_path}")
    all_stats = []
    records = []
    for kind, parent in SOURCES.items():
        group_stats = []
        for source in sorted(parent.iterdir()):
            if not (source / "meta/info.json").is_file():
                continue
            destination = args.output / kind / source.name
            record = prepare_dataset(source, destination, kind)
            files = sorted((source / "data").rglob("*.parquet"))
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                stats = list(executor.map(compute_episode, files))
            if len(stats) != record["episodes"]:
                raise ValueError(f"Parquet count mismatch: {source}")
            group_stats.extend(stats)
            dataset_stats = openpi_stats(aggregate_stats(stats))
            write_json(
                destination / "assets/lerobot_episode_stats/norm_stats.json",
                dataset_stats,
            )
            records.append(record)
            print(json.dumps(record), flush=True)
        aggregate = aggregate_stats(group_stats)
        write_json(
            args.output / "assets" / kind / "norm_stats.json", openpi_stats(aggregate)
        )
        if kind == "sft":
            reference = json.loads(REFERENCE.read_text())
            # Refuse to prepare training if the algorithm does not reproduce
            # every reference statistic exactly after LeRobot's float32 cast.
            for key, st in openpi_stats(aggregate)["norm_stats"].items():
                for name, values in st.items():
                    np.testing.assert_array_equal(
                        values, reference["norm_stats"][key][name]
                    )
            print("Reference mean/std/q01/q99: exact float32 match", flush=True)
        all_stats.extend(group_stats)
    norm_path = args.output / "assets/lerobot_episode_stats/norm_stats.json"
    write_json(norm_path, openpi_stats(aggregate_stats(all_stats)))
    import lerobot.datasets.compute_stats as stats_module

    manifest = {
        "datasets": records,
        "episodes": sum(row["episodes"] for row in records),
        "frames": sum(row["frames"] for row in records),
        "norm_stats_path": str(norm_path),
        "norm_stats_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "reference": str(REFERENCE),
        "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "sft_reproduces_reference_exactly_float32": True,
        "method": (
            "LeRobot get_feature_stats per episode, axis=0, float32 raw frames, "
            "5000-bin histograms; aggregate_stats across all episodes. "
            "q01/q99 are episode quantiles weighted by episode frame count. "
            "Mean and pooled variance are count weighted. Cast to float32 last. "
            "No action-window weighting and no pooled-frame quantiles."
        ),
        "statistics_implementation": stats_module.__file__,
        "statistics_implementation_sha256": hashlib.sha256(
            Path(stats_module.__file__).read_bytes()
        ).hexdigest(),
        "rollout_outcome_source": "meta/rollout_provenance.jsonl: source_episode_success",
    }
    write_json(manifest_path, manifest)
    print(
        f"Prepared {manifest['episodes']} episodes, {manifest['frames']} frames",
        flush=True,
    )


if __name__ == "__main__":
    main()
