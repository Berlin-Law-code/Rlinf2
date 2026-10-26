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

"""Host-cache cleanup must preserve live tensors and pending-copy ordering."""

from types import SimpleNamespace

import pytest
import torch

import rlinf.utils.utils as memory_utils


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("release", [False, True])
def test_host_cleanup_is_opt_in_and_follows_device_synchronization(
    monkeypatch, initialized, release
):
    calls = []
    monkeypatch.setattr(
        memory_utils.Worker,
        "torch_platform",
        SimpleNamespace(
            synchronize=lambda: calls.append("sync"),
            ipc_collect=lambda: calls.append("ipc"),
            empty_cache=lambda: calls.append("device"),
        ),
    )
    monkeypatch.setattr(memory_utils.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(
        torch._C, "_host_emptyCache", lambda: calls.append("host"), raising=False
    )
    memory_utils.clear_memory(clear_host_cache=release)
    assert calls == ["sync", "gc", "ipc", "device"] + (
        ["host"] if release and initialized else []
    )


def test_host_cleanup_requires_sync_and_tolerates_an_older_torch(monkeypatch):
    with pytest.raises(ValueError, match="requires synchronized"):
        memory_utils.clear_memory(sync=False, clear_host_cache=True)
    monkeypatch.setattr(
        memory_utils.Worker,
        "torch_platform",
        SimpleNamespace(
            synchronize=lambda: None,
            ipc_collect=lambda: None,
            empty_cache=lambda: None,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.delattr(torch._C, "_host_emptyCache", raising=False)
    memory_utils.clear_memory(clear_host_cache=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch._C, "_host_emptyCache"),
    reason="CUDA host-cache API required",
)
def test_cuda_host_cleanup_releases_unused_blocks_but_preserves_live_copies(
    monkeypatch,
):
    monkeypatch.setattr(memory_utils.Worker, "torch_platform", torch.cuda)
    torch.cuda.set_device(0)
    memory_utils.clear_memory(clear_host_cache=True)
    source = torch.full((4 * 1024 * 1024,), 0.25, dtype=torch.float32, device="cuda")
    live = source.to("cpu", non_blocking=True)
    assert live.is_pinned()
    for _ in range(3):
        larger = torch.full((8 * 1024 * 1024,), 0.5, dtype=torch.float32, device="cuda")
        discarded = larger.to("cpu", non_blocking=True)
        assert discarded.is_pinned()
        del larger, discarded
        # Finish outstanding transfers before releasing unused pinned blocks.
        memory_utils.clear_memory(clear_host_cache=True)
        torch.testing.assert_close(live, torch.full_like(live, 0.25), rtol=0, atol=0)
        stats = torch.cuda.memory.host_memory_stats()
        assert stats["allocated_bytes.current"] == live.numel() * live.element_size()
    del live, source
    memory_utils.clear_memory(clear_host_cache=True)
    assert torch.cuda.memory.host_memory_stats()["allocated_bytes.current"] == 0
