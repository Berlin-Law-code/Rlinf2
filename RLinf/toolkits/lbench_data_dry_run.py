#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate the LBench LeRobot/OpenPI SFT data contract without a model.

The dataset *storage* version must match the LeRobot runtime installed next to
OpenPI.  The current ``rlinf-openpi==0.1.1`` installation resolves to
``lerobot==0.3.3``, whose native on-disk format is LeRobot v2.1.  Newer
LeRobot releases use v3.0.  Feature names such as
``observation.images.obs_camera`` are independent of that storage version.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = {
    "observation.state": (8,),
    "observation.images.obs_camera": None,
    "observation.images.obs_camera_2": None,
    "observation.images.realsense": None,
    "action": (8,),
}
CORE_FEATURES = {
    "timestamp": ("float32", (1,)),
    "frame_index": ("int64", (1,)),
    "episode_index": ("int64", (1,)),
    "index": ("int64", (1,)),
    "task_index": ("int64", (1,)),
}

_V21_METADATA_FILES = (
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
)
_V30_METADATA_FILES = ("meta/tasks.parquet",)


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(value).shape)


def validate_sample(sample: dict[str, Any], horizon: int = 50) -> dict[str, Any]:
    state = np.asarray(sample["observation.state"], dtype=np.float32)
    actions = np.asarray(sample["action"], dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"observation.state must be finite [8], got {state.shape}")
    if actions.shape != (horizon, 8) or not np.isfinite(actions).all():
        raise ValueError(
            f"action chunk must be finite [{horizon}, 8], got {actions.shape}"
        )
    for name, gripper in (
        ("observation.state", state[-1:]),
        ("action", actions[:, -1]),
    ):
        if (gripper < -1.0e-4).any() or (gripper > 1.0 + 1.0e-4).any():
            raise ValueError(
                f"{name} gripper channel must use 0=closed / 1=open in [0, 1]"
            )
    image_shapes = {}
    for key in FEATURES:
        if not key.startswith("observation.images."):
            continue
        image = np.asarray(sample[key])
        if image.ndim != 3 or 3 not in (image.shape[0], image.shape[-1]):
            raise ValueError(f"{key} must be CHW or HWC RGB, got {image.shape}")
        if not (
            np.issubdtype(image.dtype, np.number)
            or np.issubdtype(image.dtype, np.bool_)
        ):
            raise TypeError(f"{key} must have a numeric dtype, got {image.dtype}")
        if np.issubdtype(image.dtype, np.floating) and not np.isfinite(image).all():
            raise ValueError(f"{key} contains NaN or Inf")
        image_shapes[key] = _shape(image)
    task = sample.get("task", sample.get("prompt"))
    if isinstance(task, np.ndarray) and task.ndim == 0:
        task = task.item()
    if isinstance(task, (bytes, np.bytes_)):
        task = task.decode("utf-8")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("sample must contain a non-empty string task/prompt")
    return {
        "state": list(state.shape),
        "actions": list(actions.shape),
        "images": {key: list(value) for key, value in image_shapes.items()},
        "prompt": task.strip(),
    }


def validate_norm_stats(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("norm_stats.json must contain a JSON object")
    # openpi.shared.normalize.save() serializes NormStatsDict under this
    # top-level field.  Keep accepting the legacy unwrapped mapping as well,
    # since both representations have been produced by our data tooling.
    if "norm_stats" in payload:
        payload = payload["norm_stats"]
        if not isinstance(payload, dict):
            raise ValueError("norm_stats.json field 'norm_stats' must be an object")
    result = {}
    for key, dim in (("state", 8), ("actions", 8)):
        if key not in payload:
            raise ValueError(f"norm_stats.json is missing {key!r}")
        stats = payload[key]
        if not isinstance(stats, dict):
            raise ValueError(f"norm stats {key} must be an object")
        # Pi0.5 uses quantile normalization, while some OpenPI utilities also
        # consume mean/std. Requiring the complete serialized NormStats schema
        # prevents a preflight pass followed by a runtime normalization error.
        for name in ("mean", "std", "q01", "q99"):
            if name not in stats:
                raise ValueError(f"norm stats {key} is missing {name!r}")
            array = np.asarray(stats[name])
            if array.shape != (dim,):
                raise ValueError(
                    f"norm stats {key}.{name} must be {dim}-D, got {array.shape}"
                )
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"norm stats {key}.{name} must be numeric")
            if not np.isfinite(array).all():
                raise ValueError(f"norm stats {key}.{name} contains NaN or Inf")
        if (np.asarray(stats["std"], dtype=np.float64) < 0).any():
            raise ValueError(f"norm stats {key}.std must be non-negative")
        if (
            np.asarray(stats["q01"], dtype=np.float64)
            > np.asarray(stats["q99"], dtype=np.float64)
        ).any():
            raise ValueError(f"norm stats {key} has q01 greater than q99")
        q01 = np.asarray(stats["q01"], dtype=np.float64)
        q99 = np.asarray(stats["q99"], dtype=np.float64)
        if q01[-1] < -1.0e-4 or q99[-1] > 1.0 + 1.0e-4:
            raise ValueError(
                f"norm stats {key} gripper channel is outside the canonical "
                "0=closed / 1=open range"
            )
        result[key] = [dim]
    return result


def _dataset_root(repo_id: str) -> Path:
    path = Path(repo_id).expanduser()
    if (path / "meta" / "info.json").is_file():
        return path.resolve()
    home = Path(
        os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")
    ).expanduser()
    return (home / repo_id).resolve()


def _storage_format(codebase_version: Any) -> str:
    """Normalize a LeRobot codebase version to its on-disk format family."""
    version = str(codebase_version or "")
    if version.startswith("v2.1"):
        return "v2.1"
    if version.startswith("v3"):
        return "v3.0"
    raise ValueError(
        "LBench OpenPI SFT supports LeRobot v2.1 or v3.0 metadata, "
        f"got codebase_version={version!r}"
    )


def _load_jsonlines(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"expected a JSON object in {path} at line {line_number}"
                )
            records.append(record)
    return records


def _format_v21_path(
    template: str,
    episode_index: int,
    *,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    values: dict[str, Any] = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    try:
        path = Path(template.format(**values))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid LeRobot v2.1 path template {template!r}") from exc
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"LeRobot v2.1 path template must stay inside the dataset: {template!r}"
        )
    return path


def _validate_v21_layout(
    root: Path, info: dict[str, Any], features: dict[str, Any]
) -> None:
    """Validate every file LeRobot 0.3.3 opens before it tries Hub fallback."""
    total_episodes = int(info.get("total_episodes", 0))
    total_tasks = int(info.get("total_tasks", 0))
    chunks_size = int(info.get("chunks_size", 0))
    if chunks_size < 1:
        raise ValueError("LeRobot v2.1 metadata chunks_size must be positive")

    tasks = _load_jsonlines(root / "meta/tasks.jsonl")
    task_indices = [int(record.get("task_index", -1)) for record in tasks]
    if len(task_indices) != total_tasks or sorted(task_indices) != list(
        range(total_tasks)
    ):
        raise ValueError(
            "LeRobot v2.1 tasks.jsonl indices must be contiguous and match "
            f"total_tasks={total_tasks}; got {task_indices[:10]}"
        )
    declared_task_names = [str(record.get("task", "")).strip() for record in tasks]
    if any(not task for task in declared_task_names):
        raise ValueError("LeRobot v2.1 tasks.jsonl contains an empty task prompt")
    if len(set(declared_task_names)) != len(declared_task_names):
        raise ValueError("LeRobot v2.1 tasks.jsonl contains duplicate task prompts")
    task_names = set(declared_task_names)

    episodes = _load_jsonlines(root / "meta/episodes.jsonl")
    episode_indices = [int(record.get("episode_index", -1)) for record in episodes]
    expected_episode_indices = list(range(total_episodes))
    if (
        len(episode_indices) != total_episodes
        or sorted(episode_indices) != expected_episode_indices
    ):
        raise ValueError(
            "LeRobot v2.1 episodes.jsonl indices must be contiguous and match "
            f"total_episodes={total_episodes}; got {episode_indices[:10]}"
        )
    invalid_lengths = [
        (int(record.get("episode_index", -1)), int(record.get("length", 0)))
        for record in episodes
        if int(record.get("length", 0)) < 1
    ]
    if invalid_lengths:
        raise ValueError(
            "LeRobot v2.1 episode lengths must be positive; got "
            f"{invalid_lengths[:10]}"
        )
    frame_total = sum(int(record.get("length", 0)) for record in episodes)
    if frame_total != int(info.get("total_frames", -1)):
        raise ValueError(
            "LeRobot v2.1 episode lengths do not match total_frames: "
            f"{frame_total} != {info.get('total_frames')}"
        )
    episode_task_names: set[str] = set()
    for record in episodes:
        record_tasks = record.get("tasks")
        if not isinstance(record_tasks, list) or len(record_tasks) != 1:
            raise ValueError(
                "LBench LeRobot v2.1 episode metadata must contain exactly one task"
            )
        task_name = str(record_tasks[0]).strip()
        if not task_name:
            raise ValueError(
                "LBench LeRobot v2.1 episode metadata contains an empty task"
            )
        episode_task_names.add(task_name)
    unknown_tasks = sorted(episode_task_names - task_names)
    missing_tasks = sorted(task_names - episode_task_names)
    if unknown_tasks or missing_tasks:
        raise ValueError(
            "LeRobot v2.1 task metadata and episode task coverage disagree: "
            f"unknown={unknown_tasks}, unused={missing_tasks}"
        )

    episode_stats = _load_jsonlines(root / "meta/episodes_stats.jsonl")
    stats_indices = [int(record.get("episode_index", -1)) for record in episode_stats]
    if (
        len(stats_indices) != total_episodes
        or sorted(stats_indices) != expected_episode_indices
    ):
        raise ValueError(
            "LeRobot v2.1 episodes_stats.jsonl must contain exactly one record "
            "per episode"
        )
    malformed_stats = [
        int(record.get("episode_index", -1))
        for record in episode_stats
        if not isinstance(record.get("stats"), Mapping)
    ]
    if malformed_stats:
        raise ValueError(
            "LeRobot v2.1 episode statistics must be mappings; malformed "
            f"episode indices (first 10): {malformed_stats[:10]}"
        )

    data_template = info.get("data_path")
    if not isinstance(data_template, str) or not data_template:
        raise ValueError("LeRobot v2.1 info.json must define data_path")
    missing_data = []
    for episode_index in expected_episode_indices:
        relative = _format_v21_path(
            data_template, episode_index, chunks_size=chunks_size
        )
        if not (root / relative).is_file():
            missing_data.append(str(relative))
            if len(missing_data) == 10:
                break
    if missing_data:
        raise FileNotFoundError(
            "LeRobot v2.1 dataset is missing episode parquet files (first up "
            f"to 10): {missing_data}"
        )

    video_keys = [
        key for key, feature in features.items() if feature.get("dtype") == "video"
    ]
    if not video_keys:
        return
    video_template = info.get("video_path")
    if not isinstance(video_template, str) or not video_template:
        raise ValueError(
            "LeRobot v2.1 metadata declares video features but has no video_path"
        )
    missing_videos = []
    for episode_index in expected_episode_indices:
        for video_key in video_keys:
            relative = _format_v21_path(
                video_template,
                episode_index,
                chunks_size=chunks_size,
                video_key=video_key,
            )
            if not (root / relative).is_file():
                missing_videos.append(str(relative))
                if len(missing_videos) == 10:
                    break
        if len(missing_videos) == 10:
            break
    if missing_videos:
        raise FileNotFoundError(
            "LeRobot v2.1 dataset is missing episode video files (first up "
            f"to 10): {missing_videos}"
        )


def _validate_layout(root: Path, storage_format: str, features: dict[str, Any]) -> None:
    """Check files that the corresponding LeRobot loader will open."""
    required = _V21_METADATA_FILES if storage_format == "v2.1" else _V30_METADATA_FILES
    missing = [relative for relative in required if not (root / relative).is_file()]
    if storage_format == "v3.0" and not any(
        (root / "meta" / "episodes").rglob("*.parquet")
    ):
        missing.append("meta/episodes/**/*.parquet")
    if not any((root / "data").rglob("*.parquet")):
        missing.append("data/**/*.parquet")
    if missing:
        raise FileNotFoundError(
            f"LeRobot {storage_format} dataset is incomplete under {root}; "
            f"missing {missing}"
        )

    if storage_format == "v2.1":
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        _validate_v21_layout(root, info, features)

    video_keys = [
        key for key, feature in features.items() if feature.get("dtype") == "video"
    ]
    if video_keys and not any((root / "videos").rglob("*.mp4")):
        raise FileNotFoundError(
            f"LeRobot metadata declares video features {video_keys}, but no MP4 "
            f"files exist under {root / 'videos'}"
        )


def _dataset_task_prompts(root: Path, storage_format: str) -> list[str]:
    """Return task prompts in task-index order for either supported layout."""

    if storage_format == "v2.1":
        records = _load_jsonlines(root / "meta/tasks.jsonl")
    else:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyarrow is required to inspect LeRobot v3 task metadata"
            ) from exc
        records = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    try:
        indexed = sorted(
            (int(record["task_index"]), str(record["task"]).strip())
            for record in records
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid LeRobot task metadata") from exc
    indices = [index for index, _prompt in indexed]
    prompts = [prompt for _index, prompt in indexed]
    if indices != list(range(len(indexed))):
        raise ValueError("LeRobot task indices must be contiguous from zero")
    if any(not prompt for prompt in prompts) or len(set(prompts)) != len(prompts):
        raise ValueError("LeRobot task prompts must be non-empty and unique")
    return prompts


def _validate_metadata(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    storage_format = _storage_format(info.get("codebase_version"))
    if int(info.get("fps", 0)) < 1:
        raise ValueError("LeRobot metadata fps must be positive")
    for key in ("total_episodes", "total_frames", "total_tasks"):
        if int(info.get(key, 0)) < 1:
            raise ValueError(f"LeRobot metadata {key} must be positive")
    features = info.get("features", {})
    if not isinstance(features, Mapping):
        raise ValueError("LeRobot metadata features must be an object")
    missing = sorted((set(FEATURES) | set(CORE_FEATURES)) - set(features))
    if missing:
        raise ValueError(f"LeRobot dataset is missing features: {missing}")
    for key, (expected_dtype, expected_shape) in CORE_FEATURES.items():
        feature = features[key]
        actual_dtype = str(feature.get("dtype", ""))
        actual_shape = tuple(feature.get("shape", ()))
        if actual_dtype != expected_dtype or actual_shape != expected_shape:
            raise ValueError(
                f"{key} metadata must be {expected_dtype}{expected_shape}, got "
                f"{actual_dtype}{actual_shape}"
            )
    for key, expected in FEATURES.items():
        dtype = str(features[key].get("dtype", ""))
        if key.startswith("observation.images."):
            if dtype not in {"image", "video"}:
                raise ValueError(
                    f"{key} metadata dtype must be image or video, got {dtype!r}"
                )
        elif dtype != "float32":
            raise ValueError(f"{key} metadata dtype must be float32, got {dtype!r}")
        if expected is None:
            actual = tuple(features[key].get("shape", ()))
            if len(actual) != 3 or 3 not in (actual[0], actual[-1]):
                raise ValueError(
                    f"{key} metadata must describe CHW or HWC RGB, got {actual}"
                )
            continue
        actual = tuple(features[key].get("shape", ()))
        if actual != expected:
            raise ValueError(f"{key} metadata shape {actual} != {expected}")
    _validate_layout(root, storage_format, features)
    task_prompts = _dataset_task_prompts(root, storage_format)
    if len(task_prompts) != int(info["total_tasks"]):
        raise ValueError(
            "LeRobot task metadata count does not match total_tasks: "
            f"{len(task_prompts)} != {info['total_tasks']}"
        )
    return info


def _v21_task_episode_counts(root: Path) -> dict[str, int]:
    """Count episode membership per task for an already validated v2.1 dataset."""
    counts = {
        str(record["task"]).strip(): 0
        for record in _load_jsonlines(root / "meta/tasks.jsonl")
    }
    for episode in _load_jsonlines(root / "meta/episodes.jsonl"):
        for task in {str(value).strip() for value in episode["tasks"]}:
            counts[task] += 1
    return counts


def _import_lerobot_dataset_module():
    """Import the available loader without hiding a missing dependency."""

    for name in (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # Continue only when this candidate module/package itself is
            # absent. If its import failed because torchcodec, torchvision, or
            # another dependency is missing, preserve that actionable error.
            missing = str(exc.name or "")
            if not (name == missing or name.startswith(f"{missing}.")):
                raise
    raise ModuleNotFoundError("Cannot import a LeRobot dataset loader")


def _lerobot_runtime() -> dict[str, str]:
    """Return the installed LeRobot package and native storage versions."""
    try:
        package_version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ModuleNotFoundError(
            "LeRobot is not installed in this Python environment"
        ) from exc

    module = _import_lerobot_dataset_module()

    codebase_version = getattr(module, "CODEBASE_VERSION", None)
    if codebase_version is None:
        try:
            metadata_module = importlib.import_module(
                "lerobot.datasets.dataset_metadata"
            )
            codebase_version = getattr(metadata_module, "CODEBASE_VERSION")
        except (ModuleNotFoundError, AttributeError) as exc:
            raise RuntimeError(
                "Cannot determine the installed LeRobot storage version"
            ) from exc
    return {
        "package_version": package_version,
        "codebase_version": str(codebase_version),
        "storage_format": _storage_format(codebase_version),
    }


def validate_lerobot_runtime(info: dict[str, Any]) -> dict[str, str]:
    """Reject a dataset/runtime storage mismatch before LeRobot contacts HF Hub."""
    dataset_format = _storage_format(info.get("codebase_version"))
    runtime = _lerobot_runtime()
    if runtime["storage_format"] != dataset_format:
        raise ValueError(
            "LeRobot dataset/runtime mismatch: dataset uses "
            f"{dataset_format}, but installed lerobot=={runtime['package_version']} "
            f"loads {runtime['storage_format']} metadata. Use a physically "
            "matching dataset copy; the documented rlinf-openpi==0.1.1 setup "
            "currently resolves to LeRobot v2.1. Do not rename metadata files "
            "or only edit info.json."
        )
    return runtime


def _sample_indices(length: int, count: int) -> list[int]:
    if length < 1:
        return []
    count = max(1, min(int(count), length))
    if count == 1:
        return [0]
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def _real_samples(
    repo_id: str, root: Path, fps: int, sample_count: int
) -> tuple[int, list[dict[str, Any]]]:
    LeRobotDataset = _import_lerobot_dataset_module().LeRobotDataset

    if fps < 1:
        raise ValueError(f"LeRobot dataset fps must be positive, got {fps}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        delta_timestamps={"action": [step / fps for step in range(50)]},
    )
    if len(dataset) < 1:
        raise ValueError("LeRobot dataset is empty")
    reports = []
    for index in _sample_indices(len(dataset), sample_count):
        sample = dict(dataset[index])
        if "task" not in sample and "prompt" not in sample:
            task_index = int(np.asarray(sample["task_index"]).reshape(-1)[0])
            sample["task"] = dataset.meta.tasks[task_index]
        report = validate_sample(sample)
        report["index"] = index
        reports.append(report)
    return len(dataset), reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id")
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="Number of evenly spaced frames to decode and validate (default: 3).",
    )
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")
    if args.mock:
        sample = {
            "observation.state": np.zeros(8, dtype=np.float32),
            "observation.images.obs_camera": np.zeros((3, 32, 32), dtype=np.uint8),
            "observation.images.obs_camera_2": np.zeros((3, 32, 32), dtype=np.uint8),
            "observation.images.realsense": np.zeros((3, 32, 32), dtype=np.uint8),
            "action": np.zeros((50, 8), dtype=np.float32),
            "task": "mock LBench task",
        }
        report = {"mode": "mock", "sample": validate_sample(sample)}
    else:
        if not args.repo_id:
            parser.error("--repo-id is required without --mock")
        root = _dataset_root(args.repo_id)
        info = _validate_metadata(root)
        runtime = validate_lerobot_runtime(info)
        dataset_length, samples = _real_samples(
            args.repo_id,
            root,
            int(info.get("fps", 0)),
            args.sample_count,
        )
        report = {
            "mode": "real",
            "root": str(root),
            "codebase_version": info.get("codebase_version"),
            "storage_format": _storage_format(info.get("codebase_version")),
            "lerobot_runtime": runtime,
            "dataset_length": dataset_length,
            "task_prompts": _dataset_task_prompts(
                root, _storage_format(info.get("codebase_version"))
            ),
            "task_episode_counts": (
                _v21_task_episode_counts(root)
                if _storage_format(info.get("codebase_version")) == "v2.1"
                else None
            ),
            # Retain the old field for callers that consumed the first report.
            "sample": samples[0],
            "samples": samples,
        }
    if args.norm_stats is not None:
        if not args.norm_stats.is_file():
            raise FileNotFoundError(args.norm_stats)
        report["norm_stats"] = validate_norm_stats(args.norm_stats)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
