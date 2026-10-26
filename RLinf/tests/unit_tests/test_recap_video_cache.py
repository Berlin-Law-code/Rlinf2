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

"""Check reader reuse, eviction, process isolation and timestamp handling."""

import os
import weakref
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.datasets.recap.video_dataset import (
    CachedVideoLeRobotDataset,
    LeRobotDataset,
)


@pytest.fixture
def cached_dataset(monkeypatch, tmp_path):
    decoders = pytest.importorskip("torchcodec.decoders")

    class Decoder:
        opens = 0
        live = weakref.WeakSet()

        def __init__(self, path, **kwargs):
            type(self).opens += 1
            type(self).live.add(self)
            self.metadata = SimpleNamespace(average_fps=30)

        def get_frames_at(self, indices):
            return SimpleNamespace(
                data=torch.tensor(indices, dtype=torch.uint8)
                .reshape(-1, 1, 1, 1)
                .expand(-1, 3, 2, 2),
                pts_seconds=torch.tensor(indices, dtype=torch.float64) / 30,
            )

    monkeypatch.setattr(decoders, "VideoDecoder", Decoder)
    dataset = CachedVideoLeRobotDataset.__new__(CachedVideoLeRobotDataset)
    dataset.root = tmp_path
    dataset.meta = SimpleNamespace(
        get_video_file_path=lambda episode, camera: Path(f"{episode}_{camera}.mp4")
    )
    dataset.video_backend = "torchcodec"
    dataset.tolerance_s = 1e-4
    dataset.video_decoder_cache_size = 3
    dataset._video_decoders = OrderedDict()
    dataset._video_decoder_pid = None
    return dataset, Decoder


def test_cached_pixels_keep_order_duplicates_and_float_range(cached_dataset):
    dataset, decoder = cached_dataset
    indices = [3, 1, 3, 2]
    frames = dataset._query_videos({"cam": [i / 30 for i in indices]}, 0)["cam"]
    expected = (
        torch.tensor(indices, dtype=torch.float32)
        .reshape(-1, 1, 1, 1)
        .expand(-1, 3, 2, 2)
        / 255
    )
    torch.testing.assert_close(frames, expected, rtol=0, atol=0)
    assert dataset._query_videos({"cam": [0.0]}, 0)["cam"].shape == (3, 2, 2)
    assert decoder.opens == 1


def test_reader_eviction_releases_resources_and_reuses_recent_camera(cached_dataset):
    dataset, decoder = cached_dataset
    for camera in ("a", "b", "c", "a", "d", "a"):
        dataset._query_videos({camera: [0.0]}, 0)
    assert decoder.opens == 4
    assert len(decoder.live) == 3
    for episode in range(100):
        dataset._query_videos({"a": [0.0]}, episode)
        assert len(dataset._video_decoders) <= 3
        assert len(decoder.live) <= 3


def test_worker_state_never_serializes_or_reuses_parent_readers(cached_dataset):
    dataset, decoder = cached_dataset
    dataset._query_videos({"a": [0.0]}, 0)
    state = dataset.__getstate__()
    assert not state["_video_decoders"]
    assert state["_video_decoder_pid"] is None
    assert len(dataset._video_decoders) == 1
    dataset._video_decoder_pid = os.getpid() - 1
    dataset._query_videos({"a": [0.0]}, 0)
    assert decoder.opens == 2
    assert len(decoder.live) == 1


def test_cached_reader_still_rejects_wrong_timestamps(cached_dataset):
    dataset, _ = cached_dataset
    with pytest.raises(AssertionError, match="timestamps exceed tolerance"):
        dataset._query_videos({"a": [0.05]}, 0)


@pytest.mark.parametrize("cache_size,backend", [(0, "torchcodec"), (3, "pyav")])
def test_disabled_or_other_backend_keeps_original_reader(
    cached_dataset, monkeypatch, cache_size, backend
):
    dataset, decoder = cached_dataset
    dataset.video_decoder_cache_size = cache_size
    dataset.video_backend = backend
    sentinel = object()
    monkeypatch.setattr(
        LeRobotDataset, "_query_videos", lambda *args, **kwargs: sentinel
    )
    assert dataset._query_videos({"a": [0.0]}, 0) is sentinel
    assert decoder.opens == 0


def test_negative_cache_size_fails_before_opening_dataset():
    with pytest.raises(ValueError, match="nonnegative"):
        CachedVideoLeRobotDataset("unused", video_decoder_cache_size=-1)
