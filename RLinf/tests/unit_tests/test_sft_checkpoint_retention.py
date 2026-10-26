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

"""Exercise SFT checkpoint retention without starting distributed workers."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.runners.sft_runner import _CHECKPOINT_COMPLETE_MARKER, SFTRunner


class _Handle:
    def __init__(self, callback):
        self.callback = callback

    def wait(self):
        return self.callback()


class _Actor:
    def __init__(self, before_save=None, *, fail=False):
        self.before_save = before_save
        self.fail = fail

    def get_max_steps_per_epoch(self):
        return _Handle(lambda: [100])

    def save_checkpoint(self, path: str, _step: int):
        def finish():
            if self.before_save is not None:
                self.before_save()
            if self.fail:
                raise RuntimeError("actor save failed")
            (Path(path) / "weights.pt").write_bytes(b"saved")

        return _Handle(finish)


def _runner(tmp_path, monkeypatch, *, limit=0, step=1000, actor=None):
    monkeypatch.setattr("rlinf.runners.sft_runner.MetricLogger", lambda _cfg: None)
    monkeypatch.setattr("rlinf.runners.sft_runner.ScopedTimer", lambda **_kwargs: None)
    cfg = OmegaConf.create(
        {
            "runner": {
                "max_steps": 30000,
                "max_saved_checkpoints": limit,
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "run",
                },
            }
        }
    )
    runner = SFTRunner(cfg, actor or _Actor())
    runner.global_step = step
    return runner


def _checkpoint(root: Path, name: str, *, complete=True) -> Path:
    path = root / name
    (path / "actor").mkdir(parents=True)
    (path / "actor" / "weights.pt").write_bytes(b"existing")
    if complete:
        (path / _CHECKPOINT_COMPLETE_MARKER).write_text("complete\n")
    return path


def test_retention_waits_for_save_and_keeps_newest_numeric_checkpoints(
    tmp_path, monkeypatch, caplog
):
    root = tmp_path / "run" / "checkpoints"
    old = _checkpoint(root, "global_step_2")
    kept = _checkpoint(root, "global_step_10")
    current = root / "global_step_1000"

    def before_save():
        assert old.is_dir()
        assert kept.is_dir()
        assert not (current / _CHECKPOINT_COMPLETE_MARKER).exists()

    runner = _runner(
        tmp_path, monkeypatch, limit=2, actor=_Actor(before_save=before_save)
    )
    with caplog.at_level("INFO", logger="rlinf.runners.sft_runner"):
        runner._save_checkpoint()

    assert not old.exists()
    assert kept.is_dir()
    assert (current / _CHECKPOINT_COMPLETE_MARKER).read_text() == "1000\n"
    assert not list(current.glob("*.tmp"))
    assert "Saved checkpoint to" in caplog.text
    assert "Pruned checkpoint" in caplog.text


def test_retention_protects_incomplete_unrelated_and_symlink_paths(
    tmp_path, monkeypatch
):
    root = tmp_path / "run" / "checkpoints"
    old = _checkpoint(root, "global_step_1")
    protected = [
        _checkpoint(root, "global_step_2", complete=False),
        _checkpoint(root, "best_model"),
        _checkpoint(root, "global_step_3_backup"),
        _checkpoint(root, "notes"),
        _checkpoint(tmp_path / "other_run" / "checkpoints", "global_step_1"),
    ]
    outside = _checkpoint(tmp_path / "outside", "global_step_4")
    symlink = root / "global_step_4"
    symlink.symlink_to(outside, target_is_directory=True)
    symlink_marker = _checkpoint(root, "global_step_5", complete=False)
    (symlink_marker / _CHECKPOINT_COMPLETE_MARKER).symlink_to(
        outside / _CHECKPOINT_COMPLETE_MARKER
    )
    regular_file = root / "global_step_6"
    regular_file.write_text("unrelated")
    runner = _runner(tmp_path, monkeypatch, limit=1)

    runner._save_checkpoint()

    assert not old.exists()
    assert all(path.is_dir() for path in protected)
    assert symlink.is_symlink()
    assert outside.is_dir()
    assert symlink_marker.is_dir()
    assert regular_file.read_text() == "unrelated"
    assert (root / "global_step_1000" / _CHECKPOINT_COMPLETE_MARKER).is_file()


@pytest.mark.parametrize("rewriting", [False, True])
def test_failed_actor_save_does_not_mark_or_prune(tmp_path, monkeypatch, rewriting):
    root = tmp_path / "run" / "checkpoints"
    old = _checkpoint(root, "global_step_1")
    current = root / "global_step_1000"
    if rewriting:
        _checkpoint(root, current.name)
    runner = _runner(tmp_path, monkeypatch, limit=1, actor=_Actor(fail=True))

    with pytest.raises(RuntimeError, match="actor save failed"):
        runner._save_checkpoint()

    assert old.is_dir()
    assert not (current / _CHECKPOINT_COMPLETE_MARKER).exists()


def test_failed_marker_publication_does_not_prune(tmp_path, monkeypatch):
    root = tmp_path / "run" / "checkpoints"
    old = _checkpoint(root, "global_step_1")
    current = root / "global_step_1000"
    runner = _runner(tmp_path, monkeypatch, limit=1)

    def fail_replace(source, destination):
        assert Path(source).read_text() == "1000\n"
        assert not Path(destination).exists()
        raise OSError("marker rename failed")

    monkeypatch.setattr("rlinf.runners.sft_runner.os.replace", fail_replace)
    with pytest.raises(OSError, match="marker rename failed"):
        runner._save_checkpoint()

    assert old.is_dir()
    assert not (current / _CHECKPOINT_COMPLETE_MARKER).exists()
    assert not list(current.glob("*.tmp"))


def test_default_retention_is_unlimited(tmp_path, monkeypatch):
    root = tmp_path / "run" / "checkpoints"
    old = _checkpoint(root, "global_step_1")
    runner = _runner(tmp_path, monkeypatch)
    del runner.cfg.runner.max_saved_checkpoints
    runner = SFTRunner(runner.cfg, _Actor())
    runner.global_step = 1000

    runner._save_checkpoint()

    assert runner.max_saved_checkpoints == 0
    assert old.is_dir()
    assert (root / "global_step_1000" / _CHECKPOINT_COMPLETE_MARKER).is_file()


def test_retention_never_removes_current_or_future_steps(tmp_path, monkeypatch):
    root = tmp_path / "run" / "checkpoints"
    old = _checkpoint(root, "global_step_1")
    future = _checkpoint(root, "global_step_2000")
    runner = _runner(tmp_path, monkeypatch, limit=1)

    runner._save_checkpoint()

    assert not old.exists()
    assert future.is_dir()
    assert (root / "global_step_1000" / _CHECKPOINT_COMPLETE_MARKER).is_file()


def test_best_model_save_does_not_apply_retention(tmp_path, monkeypatch):
    root = tmp_path / "run" / "checkpoints"
    old = _checkpoint(root, "global_step_1")
    runner = _runner(tmp_path, monkeypatch, limit=1)

    runner._save_checkpoint(is_best=True)

    assert old.is_dir()
    assert (root / "best_model" / "actor" / "weights.pt").is_file()
    assert not (root / "best_model" / _CHECKPOINT_COMPLETE_MARKER).exists()


@pytest.mark.parametrize("limit", [-1, 1.5, True, "3"])
def test_invalid_retention_fails_before_actor_access(tmp_path, monkeypatch, limit):
    with pytest.raises(ValueError, match="nonnegative integer"):
        _runner(tmp_path, monkeypatch, limit=limit, actor=object())
