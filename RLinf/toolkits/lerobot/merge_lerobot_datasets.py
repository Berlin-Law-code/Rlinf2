#!/usr/bin/env python3
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

"""Merge LeRobot v2.1 datasets found under directories into one dataset.

Recursively discovers every sub-directory that contains a valid LeRobot layout
(``meta/info.json`` + ``meta/episodes.jsonl`` + ``data/``), re-indexes all
episodes and frames globally, copies/re-indexes episode videos, rebuilds
statistics, and writes the unified dataset to ``--output-dir``.

This tool intentionally supports the v2.1 physical layout used by the current
``rlinf-openpi==0.1.1`` environment. It rejects v3.0 datasets instead of
silently producing a mixed or incomplete layout.

Typical usage
-------------
Single run directory
    python toolkits/lerobot/merge_lerobot_datasets.py \\
        --source-dir logs/20260402-16:27:36-maniskill_ppo_cnn/maniskill \\
        --output-dir merged_data

Multiple independent run directories into one dataset
    python toolkits/lerobot/merge_lerobot_datasets.py \\
        --source-dir logs/run_a/maniskill logs/run_b/maniskill \\
        --output-dir merged_data

Dry-run (just print what would be merged, no files written)
    python toolkits/lerobot/merge_lerobot_datasets.py \\
        --source-dir logs/20260402-16:27:36-maniskill_ppo_cnn/maniskill \\
        --output-dir merged_data --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _is_lerobot_dataset(path: Path) -> bool:
    """Return True if *path* looks like a LeRobot dataset root."""
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def _discover_datasets(roots: list[Path]) -> list[Path]:
    """Walk each root and return every LeRobot dataset directory found.

    A directory is considered a dataset iff it contains ``meta/info.json``
    and a ``data/`` sub-directory.  The search is recursive so it handles
    layouts like ``rank_0/id_0/``, ``rank_0/id_64/``, etc.

    The root itself is included if it is a valid dataset.
    """
    found: list[Path] = []
    seen: set[Path] = set()

    def _walk(p: Path) -> None:
        # Canonical paths prevent duplicate input roots (or directory symlinks)
        # from silently duplicating every episode in the merged dataset. They
        # also break symlink cycles during recursive discovery.
        original = p
        try:
            p = p.expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError(
                f"cannot resolve LeRobot source path {original}: {exc}"
            ) from exc
        if p in seen:
            return
        seen.add(p)
        if _is_lerobot_dataset(p):
            found.append(p)
            # Don't recurse into sub-datasets of a dataset (data/ / meta/ are
            # leaf dirs, not nested datasets).
            return
        try:
            for child in sorted(p.iterdir()):
                if child.is_dir():
                    _walk(child)
        except PermissionError as exc:
            # Silently skipping an unreadable rank directory can publish a
            # syntactically valid but incomplete merged dataset.
            raise PermissionError(
                f"cannot inspect LeRobot source directory {p}"
            ) from exc

    for root in roots:
        _walk(root)

    return found


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _reindex_episode_stats(
    stats: dict,
    *,
    new_ep_idx: int,
    new_task_idx: int,
    new_frame_start: int,
    old_frame_start: int,
) -> dict:
    """Return a copy of *stats* with episode_index and index fields updated.

    The per-feature statistics (min/max/mean/std) for ``episode_index`` and
    ``index`` (global frame index) need to reflect the new global positions
    after re-indexing.  All other feature stats are carried over unchanged.

    Args:
        stats: Original stats dict from the source ``episodes_stats.jsonl`` record.
        new_ep_idx: New global episode index assigned to this episode.
        new_task_idx: New global task index assigned to this episode.
        new_frame_start: First global frame index in the merged dataset.
        old_frame_start: First global frame index in the source dataset.

    Returns:
        Updated stats dict.
    """
    import copy

    out = copy.deepcopy(stats)
    frame_offset = new_frame_start - old_frame_start

    # episode_index: all frames share the same constant new_ep_idx
    if "episode_index" in out:
        ep_s = out["episode_index"]
        count = ep_s.get("count", [1])
        out["episode_index"] = {
            "min": [new_ep_idx],
            "max": [new_ep_idx],
            "mean": [float(new_ep_idx)],
            "std": [0.0],
            "count": count,
        }

    # task_index may change when task tables from independent shards are
    # combined.  Keeping the source statistic here would make stats.json
    # disagree with the rewritten parquet column.
    if "task_index" in out:
        task_s = out["task_index"]
        out["task_index"] = {
            "min": [new_task_idx],
            "max": [new_task_idx],
            "mean": [float(new_task_idx)],
            "std": [0.0],
            "count": task_s.get("count", [1]),
        }

    # index (global frame index): shift by offset; std is unchanged (contiguous range)
    if "index" in out:
        idx_s = out["index"]
        out["index"] = {
            "min": [v + frame_offset for v in _ensure_list(idx_s.get("min", [0]))],
            "max": [v + frame_offset for v in _ensure_list(idx_s.get("max", [0]))],
            "mean": [v + frame_offset for v in _ensure_list(idx_s.get("mean", [0.0]))],
            "std": idx_s.get("std", [0.0]),
            "count": idx_s.get("count", [1]),
        }

    return out


def _ensure_list(value: object) -> list:
    """Wrap a scalar in a list if it is not already a list."""
    return value if isinstance(value, list) else [value]


def _feature_contract(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return storage-relevant feature fields, excluding codec metadata."""
    return {
        key: {name: feature.get(name) for name in ("dtype", "shape", "names")}
        for key, feature in info.get("features", {}).items()
    }


def _validate_v21_source(
    dataset_path: Path,
    info: dict[str, Any],
    reference_info: dict[str, Any] | None,
) -> None:
    """Fail before writing if one source cannot participate in a v2.1 merge."""
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v2.1"):
        raise ValueError(
            f"{dataset_path} uses LeRobot {version!r}; this merger supports the "
            "v2.1 physical layout required by the current RLinf/OpenPI stack"
        )
    core_features = {
        "timestamp": ("float32", (1,)),
        "frame_index": ("int64", (1,)),
        "episode_index": ("int64", (1,)),
        "index": ("int64", (1,)),
        "task_index": ("int64", (1,)),
    }
    features = info.get("features", {})
    if not isinstance(features, dict):
        raise ValueError(f"invalid LeRobot feature table in {dataset_path}")
    for key, (expected_dtype, expected_shape) in core_features.items():
        feature = features.get(key, {})
        if (
            feature.get("dtype") != expected_dtype
            or tuple(feature.get("shape", ())) != expected_shape
        ):
            raise ValueError(
                f"invalid LeRobot core feature {key!r} in {dataset_path}: "
                f"expected {expected_dtype}{expected_shape}, got {feature!r}"
            )
    for key in ("total_episodes", "total_frames", "total_tasks", "chunks_size"):
        if int(info.get(key, 0)) < 1:
            raise ValueError(
                f"invalid LeRobot source {dataset_path}: {key} must be positive"
            )
    if int(info.get("fps", 0)) < 1:
        raise ValueError(
            f"invalid LeRobot source {dataset_path}: fps must be positive"
        )
    data_template = info.get("data_path")
    if not isinstance(data_template, str) or not data_template:
        raise ValueError(
            f"invalid LeRobot source {dataset_path}: data_path must be a string"
        )
    required = (
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
    )
    missing = [name for name in required if not (dataset_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete LeRobot v2.1 source {dataset_path}; missing {missing}"
        )
    if reference_info is None:
        return
    for key in ("fps", "robot_type"):
        if info.get(key) != reference_info.get(key):
            raise ValueError(
                f"LeRobot source contract mismatch for {key}: "
                f"{dataset_path} has {info.get(key)!r}, expected "
                f"{reference_info.get(key)!r}"
            )
    if _feature_contract(info) != _feature_contract(reference_info):
        raise ValueError(
            f"LeRobot feature schema mismatch in source dataset {dataset_path}"
        )


def _format_v21_path(
    template: str,
    episode_index: int,
    *,
    chunks_size: int = 1000,
    **kwargs: Any,
) -> Path:
    """Format a LeRobot v2.1 episode path from one info.json template."""
    try:
        path = Path(
            template.format(
                episode_chunk=episode_index // chunks_size,
                episode_index=episode_index,
                **kwargs,
            )
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid LeRobot v2.1 path template {template!r}") from exc
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"LeRobot v2.1 path template must stay inside the dataset: {template!r}"
        )
    return path


def _aggregate_episode_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate serialized v2.1 episode statistics using weighted variance."""
    if not records:
        raise ValueError("cannot aggregate an empty episode-statistics list")
    stats_list = [record["stats"] for record in records]
    reference_keys = set(stats_list[0])
    inconsistent = [
        int(record.get("episode_index", -1))
        for record, stats in zip(records, stats_list)
        if set(stats) != reference_keys
    ]
    if inconsistent:
        raise ValueError(
            "episode statistics use inconsistent feature sets; affected "
            f"episode indices (first 10): {inconsistent[:10]}"
        )
    result: dict[str, Any] = {}
    for feature_key in sorted(reference_keys):
        feature_stats = [
            stats[feature_key] for stats in stats_list if feature_key in stats
        ]
        required = {"min", "max", "mean", "std", "count"}
        if any(not required <= set(stats) for stats in feature_stats):
            raise ValueError(
                f"incomplete episode statistics for feature {feature_key!r}"
            )
        for stats in feature_stats:
            numeric = {
                name: np.asarray(stats[name], dtype=np.float64)
                for name in required
            }
            if any(not np.isfinite(value).all() for value in numeric.values()):
                raise ValueError(
                    f"non-finite episode statistics for feature {feature_key!r}"
                )
            if np.any(numeric["std"] < 0) or np.any(
                numeric["min"] > numeric["max"]
            ):
                raise ValueError(
                    f"invalid min/max/std episode statistics for {feature_key!r}"
                )
        means = np.stack(
            [np.asarray(stats["mean"], dtype=np.float64) for stats in feature_stats]
        )
        variances = np.stack(
            [np.asarray(stats["std"], dtype=np.float64) ** 2 for stats in feature_stats]
        )
        counts = np.stack(
            [np.asarray(stats["count"], dtype=np.float64) for stats in feature_stats]
        )
        total_count = counts.sum(axis=0)
        if np.any(total_count <= 0):
            raise ValueError(
                f"non-positive aggregate count for feature {feature_key!r}"
            )
        broadcast_counts = counts
        while broadcast_counts.ndim < means.ndim:
            broadcast_counts = np.expand_dims(broadcast_counts, axis=-1)
        total_mean = (means * broadcast_counts).sum(axis=0) / total_count
        delta = means - total_mean
        total_variance = ((variances + delta**2) * broadcast_counts).sum(
            axis=0
        ) / total_count
        result[feature_key] = {
            "min": np.min(
                np.stack([np.asarray(stats["min"]) for stats in feature_stats]),
                axis=0,
            ).tolist(),
            "max": np.max(
                np.stack([np.asarray(stats["max"]) for stats in feature_stats]),
                axis=0,
            ).tolist(),
            "mean": total_mean.tolist(),
            "std": np.sqrt(total_variance).tolist(),
            "count": total_count.astype(np.int64).tolist(),
        }
    return result


def merge_lerobot_datasets(
    source_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    video_mode: str = "copy",
) -> int:
    """Merge all LeRobot datasets discovered under *source_dirs* into *output_dir*.

    Args:
        source_dirs: One or more root directories to search recursively.
        output_dir: Destination for the merged dataset.
        dry_run: If True, only print what would be done without writing any files.
        video_mode: Copy videos or hardlink immutable videos on the same filesystem.

    Returns:
        Total number of merged episodes.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if video_mode not in {"copy", "hardlink"}:
        raise ValueError("video_mode must be copy or hardlink")
    roots = [Path(d).expanduser() for d in source_dirs]
    invalid_roots = [path for path in roots if not path.is_dir()]
    if invalid_roots:
        raise FileNotFoundError(
            f"LeRobot source directories do not exist: {invalid_roots}"
        )
    output_path = Path(output_dir).expanduser()
    if not dry_run and output_path.is_symlink():
        raise FileExistsError(
            f"refusing to replace symlink output path: {output_path}"
        )
    if not dry_run and output_path.exists():
        if not output_path.is_dir():
            raise FileExistsError(
                f"refusing to replace non-directory output path: {output_path}"
            )
        if any(output_path.iterdir()):
            raise FileExistsError(
                f"refusing to merge into non-empty output directory: {output_path}"
            )

    # ------------------------------------------------------------------
    # 1. Discover all sub-datasets
    # ------------------------------------------------------------------
    datasets = _discover_datasets(roots)
    if not datasets:
        print(
            f"[merge] No LeRobot datasets found under: {[str(r) for r in roots]}",
            file=sys.stderr,
        )
        return 0

    print(f"[merge] Found {len(datasets)} dataset(s):")
    for d in datasets:
        print(f"  {d}")
    resolved_output = output_path.expanduser().resolve()
    for dataset_path in datasets:
        resolved_dataset = dataset_path.expanduser().resolve()
        if (
            resolved_output == resolved_dataset
            or resolved_dataset in resolved_output.parents
        ):
            raise ValueError(
                f"output directory must not be inside source dataset {resolved_dataset}"
            )

    # ------------------------------------------------------------------
    # 2. Collect all episodes across datasets
    # ------------------------------------------------------------------
    # Each entry: (dataset_path, episode_meta, parquet_path, video_paths)
    all_episodes: list[tuple[Path, dict[str, Any], Path, dict[str, Path]]] = []
    # Global task name → global task index
    global_tasks: dict[str, int] = {}
    reference_info: dict[str, Any] | None = None

    # episode_index (within dataset) → stats record, keyed by (ds_path, ep_idx)
    source_episode_stats: dict[tuple[Path, int], dict] = {}
    source_task_indices: dict[tuple[Path, str], int] = {}

    for ds_path in datasets:
        info_path = ds_path / "meta" / "info.json"
        episodes_path = ds_path / "meta" / "episodes.jsonl"

        with open(info_path) as f:
            info = json.load(f)
        _validate_v21_source(ds_path, info, reference_info)
        if reference_info is None:
            reference_info = info

        episodes = _read_jsonl(episodes_path)
        chunks_size = int(info.get("chunks_size", 1000))
        declared_task_rows = _read_jsonl(ds_path / "meta" / "tasks.jsonl")
        declared_task_indices = [
            int(record.get("task_index", -1)) for record in declared_task_rows
        ]
        declared_tasks = [
            str(record.get("task", "")).strip() for record in declared_task_rows
        ]
        if (
            len(declared_tasks) != int(info.get("total_tasks", -1))
            or sorted(declared_task_indices) != list(range(len(declared_tasks)))
            or any(not task for task in declared_tasks)
            or len(set(declared_tasks)) != len(declared_tasks)
        ):
            raise ValueError(
                f"invalid v2.1 task table in {ds_path}: task indices must be "
                "contiguous, prompts non-empty/unique, and count must match info.json"
            )
        declared_task_set = set(declared_tasks)
        declared_task_index_by_name = {
            task: task_index
            for task, task_index in zip(declared_tasks, declared_task_indices)
        }
        source_task_indices.update(
            {
                (ds_path, task): task_index
                for task, task_index in declared_task_index_by_name.items()
            }
        )

        episode_indices = [
            int(record.get("episode_index", -1)) for record in episodes
        ]
        expected_episode_indices = list(range(int(info.get("total_episodes", -1))))
        if sorted(episode_indices) != expected_episode_indices:
            raise ValueError(
                f"invalid v2.1 episode table in {ds_path}: indices must be "
                "contiguous and count must match info.json"
            )
        invalid_lengths = [
            (int(record.get("episode_index", -1)), int(record.get("length", -1)))
            for record in episodes
            if int(record.get("length", -1)) < 1
        ]
        if invalid_lengths:
            raise ValueError(
                f"invalid non-positive episode lengths in {ds_path}: "
                f"{invalid_lengths[:10]}"
            )
        episode_frame_total = sum(
            int(record.get("length", -1)) for record in episodes
        )
        if episode_frame_total != int(info.get("total_frames", -1)):
            raise ValueError(
                f"episode lengths in {ds_path} sum to {episode_frame_total}, "
                f"not info.json total_frames={info.get('total_frames')}"
            )

        # v2.1 metadata loading requires per-episode statistics.
        ep_stats_path = ds_path / "meta" / "episodes_stats.jsonl"
        source_stats_rows = _read_jsonl(ep_stats_path)
        source_stats_indices = [
            int(record.get("episode_index", -1)) for record in source_stats_rows
        ]
        if sorted(source_stats_indices) != expected_episode_indices:
            raise ValueError(
                f"invalid v2.1 episode stats in {ds_path}: expected exactly one "
                "record for every contiguous episode index"
            )
        for rec in source_stats_rows:
            if not isinstance(rec.get("stats"), dict):
                raise ValueError(
                    f"invalid episode statistics in {ds_path}: episode "
                    f"{rec.get('episode_index')} has no stats mapping"
                )
            source_episode_stats[(ds_path, int(rec["episode_index"]))] = rec

        features = info.get("features", {})
        video_keys = [
            key for key, feature in features.items() if feature.get("dtype") == "video"
        ]
        video_template = info.get("video_path")
        if video_keys and not video_template:
            raise ValueError(
                f"{ds_path} declares video features but info.json has no video_path"
            )

        source_episode_tasks: set[str] = set()
        for ep_meta in episodes:
            ep_idx = int(ep_meta["episode_index"])
            episode_tasks = ep_meta.get("tasks")
            if not isinstance(episode_tasks, list) or len(episode_tasks) != 1:
                raise ValueError(
                    f"{ds_path} episode {ep_idx} must contain exactly one task; "
                    f"got {episode_tasks!r}"
                )
            task = str(episode_tasks[0]).strip()
            if not task or task not in declared_task_set:
                raise ValueError(
                    f"{ds_path} episode {ep_idx} references an unknown/empty task: "
                    f"{episode_tasks[0]!r}"
                )
            source_episode_tasks.add(task)
            data_template = info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            )
            parquet_path = ds_path / _format_v21_path(
                data_template,
                ep_idx,
                chunks_size=chunks_size,
            )
            if not parquet_path.is_file():
                raise FileNotFoundError(
                    f"LeRobot episode parquet not found: {parquet_path}"
                )
            stats_key = (ds_path, ep_idx)
            if stats_key not in source_episode_stats:
                raise ValueError(
                    f"missing episode statistics for {ds_path} episode {ep_idx}"
                )

            video_paths: dict[str, Path] = {}
            for video_key in video_keys:
                relative_video = _format_v21_path(
                    video_template,
                    ep_idx,
                    chunks_size=chunks_size,
                    video_key=video_key,
                )
                source_video = ds_path / relative_video
                if not source_video.is_file():
                    raise FileNotFoundError(
                        f"LeRobot episode video not found: {source_video}"
                    )
                video_paths[video_key] = source_video

            all_episodes.append((ds_path, ep_meta, parquet_path, video_paths))

            if task not in global_tasks:
                global_tasks[task] = len(global_tasks)

        unused_tasks = sorted(declared_task_set - source_episode_tasks)
        if unused_tasks:
            raise ValueError(
                f"v2.1 task table in {ds_path} declares tasks with no episodes: "
                f"{unused_tasks}"
            )

    total_episodes = len(all_episodes)
    print(f"[merge] Total episodes to merge: {total_episodes}")
    print(
        f"[merge] Unique tasks: {len(global_tasks)} → {list(global_tasks.keys())[:5]}"
    )

    if total_episodes == 0:
        print("[merge] Nothing to merge.", file=sys.stderr)
        return 0

    if dry_run:
        print("[merge] Dry-run mode — no files written.")
        return total_episodes

    # ------------------------------------------------------------------
    # 3. Prepare an atomic staging directory
    # ------------------------------------------------------------------
    output_target = output_path
    output_target.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = tempfile.TemporaryDirectory(
        prefix=f".{output_target.name}.merge-",
        dir=str(output_target.parent),
    )
    output_path = Path(staging_directory.name)
    (output_path / "meta").mkdir(exist_ok=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Re-index and write parquet files
    # ------------------------------------------------------------------
    global_frame_index = 0
    merged_episode_metas: list[dict] = []
    merged_episode_stats: list[dict] = []
    reference_arrow_fields: dict[str, tuple[Any, bool]] | None = None
    reference_arrow_order: list[str] | None = None

    for new_ep_idx, (
        ds_path,
        ep_meta,
        parquet_path,
        source_videos,
    ) in enumerate(all_episodes):
        table = pq.read_table(parquet_path)
        n_frames = table.num_rows
        if n_frames < 1:
            raise ValueError(f"source parquet contains no frames: {parquet_path}")
        required_columns = {
            "index",
            "episode_index",
            "frame_index",
            "task_index",
            "timestamp",
        }
        missing_columns = sorted(required_columns - set(table.column_names))
        if missing_columns:
            raise ValueError(
                f"source parquet {parquet_path} lacks required columns: "
                f"{missing_columns}"
            )
        expected_data_columns = {
            key
            for key, feature in (reference_info or {}).get("features", {}).items()
            # LeRobot v2.1 stores video frames only in MP4 files. Image bytes,
            # in contrast, are embedded in the episode parquet.
            if feature.get("dtype") != "video"
        }
        actual_data_columns = set(table.column_names)
        missing_feature_columns = sorted(expected_data_columns - actual_data_columns)
        unexpected_columns = sorted(actual_data_columns - expected_data_columns)
        if missing_feature_columns or unexpected_columns:
            raise ValueError(
                f"source parquet {parquet_path} disagrees with info.json features: "
                f"missing={missing_feature_columns}, extra={unexpected_columns}"
            )
        arrow_fields = {
            field.name: (field.type, field.nullable) for field in table.schema
        }
        if reference_arrow_fields is None:
            reference_arrow_fields = arrow_fields
            reference_arrow_order = list(table.column_names)
        elif arrow_fields != reference_arrow_fields:
            mismatched_fields = sorted(
                name
                for name in set(arrow_fields) | set(reference_arrow_fields)
                if arrow_fields.get(name) != reference_arrow_fields.get(name)
            )
            raise ValueError(
                "source parquet Arrow schemas differ across shards; mismatched "
                f"fields in {parquet_path}: {mismatched_fields}"
            )
        # Normalize column order as well as types so every output parquet has a
        # uniform logical schema, even if a source writer emitted dict keys in a
        # different insertion order.
        if reference_arrow_order is None:
            raise RuntimeError("missing reference Arrow column order")
        table = table.select(reference_arrow_order)
        declared_length = int(ep_meta.get("length", -1))
        if n_frames != declared_length:
            raise ValueError(
                f"source parquet {parquet_path} has {n_frames} frames, but "
                f"episodes.jsonl declares {declared_length}"
            )

        old_ep_idx: int = ep_meta["episode_index"]
        task = str(ep_meta["tasks"][0]).strip()
        new_task_idx = global_tasks[task]
        expected_source_task_idx = source_task_indices[(ds_path, task)]

        old_index_values = [int(value) for value in table.column("index").to_pylist()]
        old_frame_start = min(old_index_values)
        if old_index_values != list(range(old_frame_start, old_frame_start + n_frames)):
            raise ValueError(
                f"source parquet {parquet_path} has a non-contiguous or "
                "out-of-order global index column"
            )
        episode_index_values = {
            int(value) for value in table.column("episode_index").to_pylist()
        }
        if episode_index_values != {old_ep_idx}:
            raise ValueError(
                f"source parquet {parquet_path} episode_index values "
                f"{sorted(episode_index_values)} do not match metadata {old_ep_idx}"
            )
        frame_index_values = [
            int(value) for value in table.column("frame_index").to_pylist()
        ]
        if frame_index_values != list(range(n_frames)):
            raise ValueError(
                f"source parquet {parquet_path} frame_index must be 0..{n_frames - 1}"
            )
        task_index_values = {
            int(value) for value in table.column("task_index").to_pylist()
        }
        if task_index_values != {expected_source_task_idx}:
            raise ValueError(
                f"source parquet {parquet_path} task_index values "
                f"{sorted(task_index_values)} do not match task {task!r} "
                f"(index {expected_source_task_idx})"
            )

        # Replace only the three index columns on the original Arrow table.
        # A pandas round-trip can change embedded image structs, list widths,
        # extension types, or schema metadata used by Hugging Face/LeRobot.
        replacements = {
            "episode_index": [new_ep_idx] * n_frames,
            "index": range(global_frame_index, global_frame_index + n_frames),
            "task_index": [new_task_idx] * n_frames,
        }
        for column_name, values in replacements.items():
            column_index = table.schema.get_field_index(column_name)
            field = table.schema.field(column_index)
            table = table.set_column(
                column_index,
                field,
                pa.array(values, type=field.type),
            )

        # Determine output chunk
        output_chunks_size = 1000
        chunk_idx = new_ep_idx // output_chunks_size
        chunk_dir = output_path / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        out_parquet = chunk_dir / f"episode_{new_ep_idx:06d}.parquet"

        pq.write_table(table, out_parquet)

        output_video_template = (reference_info or {}).get("video_path")
        for video_key, source_video in source_videos.items():
            relative_video = _format_v21_path(
                output_video_template,
                new_ep_idx,
                chunks_size=output_chunks_size,
                video_key=video_key,
            )
            output_video = output_path / relative_video
            output_video.parent.mkdir(parents=True, exist_ok=True)
            if video_mode == "hardlink":
                os.link(source_video, output_video)
            else:
                shutil.copy2(source_video, output_video)

        merged_episode_metas.append(
            {
                "episode_index": new_ep_idx,
                "tasks": [task],
                "length": n_frames,
                **{
                    k: ep_meta[k]
                    for k in ep_meta
                    if k not in {"episode_index", "tasks", "length"}
                },
            }
        )

        src_stats_rec = source_episode_stats[(ds_path, old_ep_idx)]
        new_stats = _reindex_episode_stats(
            src_stats_rec["stats"],
            new_ep_idx=new_ep_idx,
            new_task_idx=new_task_idx,
            new_frame_start=global_frame_index,
            old_frame_start=old_frame_start,
        )
        merged_episode_stats.append({"episode_index": new_ep_idx, "stats": new_stats})

        global_frame_index += n_frames

        if (new_ep_idx + 1) % 50 == 0 or (new_ep_idx + 1) == total_episodes:
            print(f"[merge] Processed {new_ep_idx + 1}/{total_episodes} episodes …")

    # ------------------------------------------------------------------
    # 5. Write meta files
    # ------------------------------------------------------------------
    total_chunks = (total_episodes + output_chunks_size - 1) // output_chunks_size
    reference_features = (reference_info or {}).get("features", {})
    output_video_keys = [
        key
        for key, feature in reference_features.items()
        if feature.get("dtype") == "video"
    ]

    # Build info.json from reference, patching totals/splits
    info_out: dict[str, Any] = dict(reference_info) if reference_info else {}
    info_out.update(
        {
            "total_episodes": total_episodes,
            "total_frames": global_frame_index,
            "total_tasks": len(global_tasks),
            "total_videos": total_episodes * len(output_video_keys),
            "total_chunks": max(1, total_chunks),
            "chunks_size": output_chunks_size,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": (
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            ),
        }
    )
    if not output_video_keys:
        info_out.pop("video_path", None)

    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(info_out, f, indent=4)

    _write_jsonl(output_path / "meta" / "episodes.jsonl", merged_episode_metas)

    sorted_tasks = sorted(global_tasks.items(), key=lambda x: x[1])
    _write_jsonl(
        output_path / "meta" / "tasks.jsonl",
        [{"task_index": idx, "task": task} for task, idx in sorted_tasks],
    )

    _write_jsonl(output_path / "meta" / "episodes_stats.jsonl", merged_episode_stats)
    with open(output_path / "meta" / "stats.json", "w") as f:
        json.dump(_aggregate_episode_stats(merged_episode_stats), f, indent=4)
    print(f"[merge] Written episodes_stats.jsonl ({len(merged_episode_stats)} records)")

    # Publish only after every data/video/meta file and the aggregate stats
    # have been written successfully. TemporaryDirectory removes the staging
    # tree automatically if an exception unwinds this function.
    if output_target.is_symlink():
        raise FileExistsError(
            "merge output changed into a symlink while staging; refusing to "
            f"overwrite: {output_target}"
        )
    if output_target.exists():
        if not output_target.is_dir() or any(output_target.iterdir()):
            raise FileExistsError(
                "merge output changed while staging; refusing to overwrite: "
                f"{output_target}"
            )
        output_target.rmdir()
    output_path.replace(output_target)
    staging_directory.cleanup()
    output_path = output_target

    print(
        f"[merge] Done: {total_episodes} episodes, {global_frame_index} frames "
        f"→ {output_path}"
    )
    return total_episodes


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge LeRobot v2.1 datasets under one or more directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source-dir",
        nargs="+",
        required=True,
        metavar="DIR",
        help=(
            "Root directory (or directories) to search for LeRobot datasets. "
            "All sub-directories with a valid meta/info.json are included."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Output directory for the merged dataset.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered datasets and episode counts without writing any files.",
    )
    p.add_argument(
        "--video-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Hardlinks reuse immutable videos on the same filesystem without copying.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    n = merge_lerobot_datasets(
        source_dirs=args.source_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        video_mode=args.video_mode,
    )
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
