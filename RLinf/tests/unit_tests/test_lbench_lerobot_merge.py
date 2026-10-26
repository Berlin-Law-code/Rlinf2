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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toolkits.lerobot.merge_lerobot_datasets import merge_lerobot_datasets

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
CAMERAS = (
    "observation.images.obs_camera",
    "observation.images.obs_camera_2",
    "observation.images.realsense",
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _constant_stats(value: list[float], count: int = 2) -> dict:
    array = np.asarray(value, dtype=np.float64)
    return {
        "min": array.tolist(),
        "max": array.tolist(),
        "mean": array.tolist(),
        "std": np.zeros_like(array).tolist(),
        "count": [count],
    }


def _make_v21_shard(root: Path, task: str, marker: int) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": ["state"],
        },
        "action": {"dtype": "float32", "shape": [8], "names": ["action"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for camera in CAMERAS:
        features[camera] = {
            "dtype": "video",
            "shape": [8, 8, 3],
            "names": ["height", "width", "channel"],
        }
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "robot_type": "franka",
        "total_episodes": 1,
        "total_frames": 2,
        "total_tasks": 1,
        "total_videos": 3,
        "total_chunks": 1,
        "chunks_size": 1000,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": task}])
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [{"episode_index": 0, "tasks": [task], "length": 2}],
    )
    episode_stats = {
        "observation.state": _constant_stats([marker] * 8),
        "action": _constant_stats([marker] * 8),
        "index": {
            "min": [0],
            "max": [1],
            "mean": [0.5],
            "std": [0.5],
            "count": [2],
        },
        "frame_index": {
            "min": [0],
            "max": [1],
            "mean": [0.5],
            "std": [0.5],
            "count": [2],
        },
        "timestamp": {
            "min": [0.0],
            "max": [1 / 30],
            "mean": [1 / 60],
            "std": [1 / 60],
            "count": [2],
        },
        "episode_index": _constant_stats([0]),
        "task_index": _constant_stats([0]),
    }
    _write_jsonl(
        root / "meta/episodes_stats.jsonl",
        [{"episode_index": 0, "stats": episode_stats}],
    )
    table = pa.table(
        {
            "index": [0, 1],
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "task_index": [0, 0],
            "timestamp": np.asarray([0.0, 1 / 30], dtype=np.float32),
            "observation.state": [[marker] * 8] * 2,
            "action": [[marker] * 8] * 2,
        }
    )
    pq.write_table(table, root / "data/chunk-000/episode_000000.parquet")
    for camera in CAMERAS:
        video = root / VIDEO_PATH.format(
            episode_chunk=0, video_key=camera, episode_index=0
        )
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"{marker}-{camera}".encode())


def _convert_camera_to_embedded_image(root: Path, camera: str, marker: int) -> None:
    """Model LeRobot v2.1's embed_images() parquet representation."""

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"][camera]["dtype"] = "image"
    info["total_videos"] -= 1
    info_path.write_text(json.dumps(info), encoding="utf-8")

    parquet_path = root / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(parquet_path)
    image_storage_type = pa.struct(
        [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
    )
    embedded = pa.array(
        [
            {"bytes": f"png-{marker}-{index}".encode(), "path": None}
            for index in range(table.num_rows)
        ],
        type=image_storage_type,
    )
    table = table.append_column(camera, embedded)
    pq.write_table(table, parquet_path)


def test_merge_hardlinks_videos_without_rewriting_sources(tmp_path):
    source = tmp_path / "source"
    _make_v21_shard(source, "task", 1)
    output = tmp_path / "merged"
    original = source / VIDEO_PATH.format(
        episode_chunk=0, video_key=CAMERAS[0], episode_index=0
    )
    before = original.read_bytes()
    assert merge_lerobot_datasets([source], output, video_mode="hardlink") == 1
    linked = output / original.relative_to(source)
    assert linked.stat().st_ino == original.stat().st_ino
    assert linked.read_bytes() == original.read_bytes() == before


def test_merge_v21_preserves_videos_stats_and_global_indices(tmp_path):
    source = tmp_path / "source"
    _make_v21_shard(source / "rank_0/id_0", "task a", 1)
    _make_v21_shard(source / "rank_1/id_0", "task b", 2)
    output = tmp_path / "merged"

    assert merge_lerobot_datasets([source], output, dry_run=True) == 2
    assert not output.exists()
    assert merge_lerobot_datasets([source], output) == 2

    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 4
    assert info["total_tasks"] == 2
    assert info["total_videos"] == 6
    assert (output / "meta/stats.json").is_file()

    for episode_index in range(2):
        table = pq.read_table(
            output / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        ).to_pydict()
        assert table["episode_index"] == [episode_index, episode_index]
        assert table["index"] == [2 * episode_index, 2 * episode_index + 1]
        for camera in CAMERAS:
            assert (
                output
                / VIDEO_PATH.format(
                    episode_chunk=0,
                    video_key=camera,
                    episode_index=episode_index,
                )
            ).is_file()

    episode_stats = [
        json.loads(line)
        for line in (output / "meta/episodes_stats.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for task_index, record in enumerate(episode_stats):
        assert record["stats"]["task_index"]["mean"] == [float(task_index)]


def test_merge_rejects_episode_with_multiple_tasks(tmp_path):
    source = tmp_path / "source/rank_0/id_0"
    _make_v21_shard(source, "task", 1)
    _write_jsonl(
        source / "meta/episodes.jsonl",
        [{"episode_index": 0, "tasks": ["task", "other"], "length": 2}],
    )

    with pytest.raises(ValueError, match="must contain exactly one task"):
        merge_lerobot_datasets([tmp_path / "source"], tmp_path / "merged")


def test_merge_rejects_v3_source_instead_of_mixing_layouts(tmp_path):
    source = tmp_path / "source/rank_0/id_0"
    _make_v21_shard(source, "task", 1)
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["codebase_version"] = "v3.0"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="supports the v2.1 physical layout"):
        merge_lerobot_datasets([tmp_path / "source"], tmp_path / "merged")


def test_merge_preserves_embedded_v21_image_bytes_and_arrow_type(tmp_path):
    source = tmp_path / "source/rank_0/id_0"
    camera = CAMERAS[0]
    _make_v21_shard(source, "task", 7)
    _convert_camera_to_embedded_image(source, camera, 7)

    output = tmp_path / "merged"
    merge_lerobot_datasets([tmp_path / "source"], output)

    source_table = pq.read_table(source / "data/chunk-000/episode_000000.parquet")
    merged_table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    assert (
        merged_table.schema.field(camera).type
        == source_table.schema.field(camera).type
    )
    assert merged_table.column(camera).to_pylist() == source_table.column(camera).to_pylist()


def test_merge_rejects_parquet_missing_declared_feature(tmp_path):
    source = tmp_path / "source/rank_0/id_0"
    _make_v21_shard(source, "task", 1)
    parquet_path = source / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(parquet_path)
    table = table.select([name for name in table.column_names if name != "action"])
    pq.write_table(table, parquet_path)

    with pytest.raises(ValueError, match="disagrees with info.json features"):
        merge_lerobot_datasets([tmp_path / "source"], tmp_path / "merged")


def test_merge_refuses_non_empty_output(tmp_path):
    source = tmp_path / "source/rank_0/id_0"
    _make_v21_shard(source, "task", 1)
    output = tmp_path / "merged"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output"):
        merge_lerobot_datasets([tmp_path / "source"], output)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_merge_deduplicates_canonical_source_roots(tmp_path):
    source = tmp_path / "source/rank_0/id_0"
    _make_v21_shard(source, "task", 1)

    assert (
        merge_lerobot_datasets(
            [tmp_path / "source", source.parent / ".." / "rank_0"],
            tmp_path / "merged",
            dry_run=True,
        )
        == 1
    )


def test_merge_refuses_to_replace_output_file(tmp_path):
    source = tmp_path / "source/rank_0/id_0"
    _make_v21_shard(source, "task", 1)
    output = tmp_path / "merged"
    output.write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-directory output"):
        merge_lerobot_datasets([tmp_path / "source"], output)
    assert output.read_text(encoding="utf-8") == "user data"


def test_merge_refuses_symlink_output(tmp_path):
    source = tmp_path / "source/rank_0/id_0"
    _make_v21_shard(source, "task", 1)
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "merged"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(FileExistsError, match="symlink output"):
        merge_lerobot_datasets([tmp_path / "source"], output)
    assert output.is_symlink()


def test_merge_rejects_arrow_schema_mismatch_between_shards(tmp_path):
    first = tmp_path / "source/rank_0/id_0"
    second = tmp_path / "source/rank_1/id_0"
    _make_v21_shard(first, "task a", 1)
    _make_v21_shard(second, "task b", 2)
    parquet_path = second / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(parquet_path)
    action_index = table.schema.get_field_index("action")
    action_field = pa.field("action", pa.list_(pa.float32()))
    table = table.set_column(
        action_index,
        action_field,
        table.column(action_index).cast(action_field.type),
    )
    pq.write_table(table, parquet_path)

    with pytest.raises(ValueError, match="Arrow schemas differ across shards"):
        merge_lerobot_datasets([tmp_path / "source"], tmp_path / "merged")


def test_merge_does_not_publish_partial_output_on_write_failure(
    tmp_path, monkeypatch
):
    source = tmp_path / "source/rank_0/id_0"
    _make_v21_shard(source, "task", 1)
    output = tmp_path / "merged"

    def fail_write(*_args, **_kwargs):
        raise OSError("simulated parquet write failure")

    monkeypatch.setattr(pq, "write_table", fail_write)
    with pytest.raises(OSError, match="simulated parquet write failure"):
        merge_lerobot_datasets([tmp_path / "source"], output)
    assert not output.exists()
