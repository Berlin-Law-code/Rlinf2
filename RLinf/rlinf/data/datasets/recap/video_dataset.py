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

"""Bounded, process-local video readers for sequential LeRobot inference."""

import os
from collections import OrderedDict

import torch

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class CachedVideoLeRobotDataset(LeRobotDataset):
    """Reuse a bounded number of TorchCodec readers without caching frames.

    Reopening a reader for every frame repeatedly decodes from a keyframe.
    Keeping a reader for each camera lets sequential access reuse decoder state.
    A zero cache size or a non-TorchCodec backend retains LeRobot's own path.
    """

    def __init__(self, *args, video_decoder_cache_size: int = 0, **kwargs):
        if video_decoder_cache_size < 0:
            raise ValueError("video_decoder_cache_size must be nonnegative")
        self.video_decoder_cache_size = video_decoder_cache_size
        self._video_decoders = OrderedDict()
        self._video_decoder_pid = None
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Native decoder objects must not be serialized into spawned workers.
        state["_video_decoders"] = OrderedDict()
        state["_video_decoder_pid"] = None
        return state

    def _get_video_decoder(self, video_path):
        from torchcodec.decoders import VideoDecoder

        pid = os.getpid()
        if self._video_decoder_pid != pid:
            self._video_decoders.clear()
            self._video_decoder_pid = pid

        key = str(video_path)
        if key in self._video_decoders:
            self._video_decoders.move_to_end(key)
            return self._video_decoders[key]

        # Evict before opening to keep the cache bounded during transitions.
        while len(self._video_decoders) >= self.video_decoder_cache_size:
            self._video_decoders.popitem(last=False)
        decoder = VideoDecoder(
            video_path, device="cpu", seek_mode="approximate", num_ffmpeg_threads=1
        )
        self._video_decoders[key] = decoder
        return decoder

    def _query_videos(self, query_timestamps, ep_idx):
        if self.video_decoder_cache_size == 0 or self.video_backend != "torchcodec":
            return super()._query_videos(query_timestamps, ep_idx)

        item = {}
        for video_key, timestamps in query_timestamps.items():
            path = self.root / self.meta.get_video_file_path(ep_idx, video_key)
            decoder = self._get_video_decoder(path)
            indices = [round(ts * decoder.metadata.average_fps) for ts in timestamps]
            frames = decoder.get_frames_at(indices=indices)

            # Match LeRobot's timestamp validation and nearest-frame selection,
            # including repeated timestamps and non-monotonic requests.
            requested = torch.tensor(timestamps, dtype=torch.float32)
            actual = frames.pts_seconds.to(dtype=torch.float32)
            distances = torch.cdist(requested[:, None], actual[:, None], p=1)
            errors, nearest = distances.min(dim=1)
            if not bool((errors < self.tolerance_s).all()):
                raise AssertionError(
                    f"Video timestamps exceed tolerance {self.tolerance_s}: "
                    f"errors={errors.tolist()}, video={path}"
                )
            pixels = frames.data.index_select(0, nearest).to(torch.float32) / 255
            item[video_key] = pixels.squeeze(0)
        return item
