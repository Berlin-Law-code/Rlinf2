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

"""CPU-only checkpoint lifecycle tests; no distributed workers or training."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import rlinf.runners.embodied_runner as runner_module
from rlinf.runners.embodied_runner import (
    _CHECKPOINT_COMPLETE_MARKER,
    EmbodiedRunner,
)


class _Actor:
    def __init__(self):
        self.before_save = None
        self.fail = False
        self.saved_steps = []

    def save_checkpoint(self, path, step):
        def finish():
            if self.before_save is not None:
                self.before_save()
            if self.fail:
                raise RuntimeError("actor save failed")
            (Path(path) / "weights.pt").write_bytes(b"saved weights")
            (Path(path) / "optimizer.pt").write_bytes(b"saved optimizer")
            self.saved_steps.append(step)

        return SimpleNamespace(wait=finish)


def _runner(tmp_path, monkeypatch, *, limit=3):
    monkeypatch.setattr(runner_module.Channel, "create", lambda _name: None)
    monkeypatch.setattr(runner_module, "MetricLogger", lambda _cfg: None)
    monkeypatch.setattr(runner_module, "ScopedTimer", lambda **kwargs: None)
    monkeypatch.setattr(
        runner_module,
        "threading",
        SimpleNamespace(Thread=lambda **kwargs: SimpleNamespace(start=lambda: None)),
    )
    cfg = OmegaConf.create(
        {
            "cluster": {},
            "runner": {
                "max_epochs": 500,
                "max_steps": 500,
                "weight_sync_interval": 1,
                "logger": {"log_path": str(tmp_path), "experiment_name": "run"},
            },
        }
    )
    if limit is not None:
        cfg.runner.max_saved_checkpoints = limit
    runner = EmbodiedRunner(cfg, _Actor(), None, None)
    runner.logger = logging.getLogger(runner_module.__name__)
    return runner


def _save(runner, step):
    runner.global_step = step
    runner._save_checkpoint()
    return (
        Path(runner.cfg.runner.logger.log_path)
        / runner.cfg.runner.logger.experiment_name
        / "checkpoints"
        / f"global_step_{step}"
    )


def _existing(path, *, complete=True):
    (path / "actor").mkdir(parents=True)
    (path / "actor" / "weights.pt").write_bytes(b"existing weights")
    if complete:
        (path / _CHECKPOINT_COMPLETE_MARKER).write_text("existing\n")
    return path


def test_fourth_successful_save_prunes_only_oldest_after_wait(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, limit=3)
    previous = [_save(runner, step) for step in (2, 10, 50)]
    current = previous[-1].parent / "global_step_100"

    def before_save():
        assert all(path.is_dir() for path in previous)
        assert not (current / _CHECKPOINT_COMPLETE_MARKER).exists()
        assert current not in runner._saved_checkpoints

    runner.actor.before_save = before_save
    _save(runner, 100)
    assert not previous[0].exists()
    assert all(path.is_dir() for path in previous[1:])
    assert (current / _CHECKPOINT_COMPLETE_MARKER).read_text() == "100\n"
    assert (current / "actor" / "optimizer.pt").is_file()
    assert not list(current.glob("*.tmp"))
    assert set(runner._saved_checkpoints) == {*previous[1:], current}


@pytest.mark.parametrize("limit", [None, 0])
def test_default_and_zero_preserve_unlimited_save_without_markers(
    tmp_path, monkeypatch, limit
):
    runner = _runner(tmp_path, monkeypatch, limit=limit)
    checkpoints = [_save(runner, step) for step in (50, 100, 150, 200)]
    assert runner.max_saved_checkpoints == 0
    assert runner._saved_checkpoints == {}
    assert all(path.is_dir() for path in checkpoints)
    assert not any(
        (path / _CHECKPOINT_COMPLETE_MARKER).exists() for path in checkpoints
    )


def test_retention_owns_only_current_instance_saves(tmp_path, monkeypatch):
    root = tmp_path / "run" / "checkpoints"
    protected = [
        _existing(root / "global_step_1"),
        _existing(root / "global_step_20", complete=False),
        _existing(root / "best_model"),
        _existing(root / "global_step_30_backup"),
        _existing(tmp_path / "sft_source" / "checkpoints" / "global_step_30000"),
    ]
    runner = _runner(tmp_path, monkeypatch, limit=1)
    first = _save(runner, 50)
    second = _save(runner, 100)
    assert not first.exists()
    assert all(path.is_dir() for path in protected)

    restarted = _runner(tmp_path, monkeypatch, limit=1)
    third = _save(restarted, 150)
    fourth = _save(restarted, 200)
    assert second.is_dir() and fourth.is_dir()
    assert not third.exists()
    assert all(path.is_dir() for path in protected)
    assert (
        protected[-1].joinpath("actor/weights.pt").read_bytes() == b"existing weights"
    )


def test_failed_save_does_not_register_mark_or_prune(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, limit=1)
    old = _save(runner, 50)
    runner.actor.fail = True
    with pytest.raises(RuntimeError, match="actor save failed"):
        _save(runner, 100)
    assert old.is_dir()
    assert set(runner._saved_checkpoints) == {old}
    assert not (old.parent / "global_step_100" / _CHECKPOINT_COMPLETE_MARKER).exists()


def test_failed_rewrite_loses_completion_and_stays_out_of_later_pruning(
    tmp_path, monkeypatch
):
    runner = _runner(tmp_path, monkeypatch, limit=2)
    old = _save(runner, 50)
    rewritten = _save(runner, 100)
    runner.max_saved_checkpoints = 1
    runner.actor.fail = True
    with pytest.raises(RuntimeError, match="actor save failed"):
        _save(runner, 100)
    assert old.is_dir() and rewritten.is_dir()
    assert not (rewritten / _CHECKPOINT_COMPLETE_MARKER).exists()
    assert rewritten not in runner._saved_checkpoints
    runner.actor.fail = False
    _save(runner, 150)
    assert not old.exists()
    assert rewritten.is_dir()


def test_failed_atomic_publication_leaves_old_checkpoints_untouched(
    tmp_path, monkeypatch
):
    runner = _runner(tmp_path, monkeypatch, limit=1)
    old = _save(runner, 50)
    current = old.parent / "global_step_100"

    def fail_replace(source, destination):
        assert Path(source).read_text() == "100\n"
        assert (current / "actor" / "optimizer.pt").is_file()
        assert not Path(destination).exists()
        raise OSError("marker rename failed")

    monkeypatch.setattr(runner_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="marker rename failed"):
        _save(runner, 100)
    assert old.is_dir()
    assert set(runner._saved_checkpoints) == {old}
    assert not (current / _CHECKPOINT_COMPLETE_MARKER).exists()
    assert not list(current.glob("*.tmp"))


@pytest.mark.parametrize(
    "replacement", ["directory_symlink", "marker_symlink", "no_marker"]
)
def test_registered_paths_must_still_be_regular_complete_directories(
    tmp_path, monkeypatch, replacement
):
    runner = _runner(tmp_path, monkeypatch, limit=1)
    old = _save(runner, 50)
    outside = _existing(tmp_path / "sft_source")
    if replacement == "directory_symlink":
        old.rename(tmp_path / "moved_checkpoint")
        old.symlink_to(outside, target_is_directory=True)
    else:
        (old / _CHECKPOINT_COMPLETE_MARKER).unlink()
        if replacement == "marker_symlink":
            (old / _CHECKPOINT_COMPLETE_MARKER).symlink_to(
                outside / _CHECKPOINT_COMPLETE_MARKER
            )
    _save(runner, 100)
    assert old.is_dir() and outside.is_dir()
    assert (outside / "actor" / "weights.pt").read_bytes() == b"existing weights"


@pytest.mark.parametrize("symlink_level", ["root", "checkpoint", "actor"])
def test_opted_in_save_rejects_symlinks_before_writing_source(
    tmp_path, monkeypatch, symlink_level
):
    runner = _runner(tmp_path, monkeypatch, limit=1)
    source = _existing(tmp_path / "sft_source")
    root = tmp_path / "run" / "checkpoints"
    target = {
        "root": root,
        "checkpoint": root / "global_step_50",
        "actor": root / "global_step_50" / "actor",
    }[symlink_level]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="checkpoint symlink"):
        _save(runner, 50)
    assert runner.actor.saved_steps == []
    assert (source / "actor" / "weights.pt").read_bytes() == b"existing weights"


def test_current_future_and_other_experiment_checkpoints_are_protected(
    tmp_path, monkeypatch
):
    runner = _runner(tmp_path, monkeypatch, limit=3)
    outside_run = _save(runner, 10)
    runner.cfg.runner.logger.experiment_name = "second_run"
    old = _save(runner, 50)
    future = _save(runner, 200)
    runner.max_saved_checkpoints = 1
    current = _save(runner, 100)
    assert current.is_dir() and future.is_dir() and outside_run.is_dir()
    assert not old.exists()


def test_prune_failure_is_logged_and_retried_after_next_success(
    tmp_path, monkeypatch, caplog
):
    runner = _runner(tmp_path, monkeypatch, limit=1)
    old = _save(runner, 50)

    def fail_remove(path):
        assert path == old
        raise PermissionError("temporary deletion failure")

    with monkeypatch.context() as patch:
        patch.setattr(runner_module.shutil, "rmtree", fail_remove)
        with caplog.at_level(logging.WARNING, logger=runner_module.__name__):
            current = _save(runner, 100)
    assert old.is_dir() and current.is_dir()
    assert "Failed to prune checkpoint" in caplog.text
    newest = _save(runner, 150)
    assert newest.is_dir() and not old.exists() and not current.exists()


@pytest.mark.parametrize("limit", [-1, 1.5, True, "3", None])
def test_invalid_limit_fails_before_channels_or_workers(limit):
    cfg = OmegaConf.create({"runner": {"max_saved_checkpoints": limit}})
    with pytest.raises(ValueError, match="nonnegative integer"):
        EmbodiedRunner(cfg, object(), None, None)
