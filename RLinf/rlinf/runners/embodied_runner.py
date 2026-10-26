# Copyright 2025 The RLinf Authors.
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

import json
import logging
import math
import os
import queue
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Union

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer

logger = logging.getLogger(__name__)
_CHECKPOINT_COMPLETE_MARKER = ".checkpoint_complete"

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor",
            "EmbodiedNFTFSDPPolicy",
            "EmbodiedSACFSDPPolicy",
            "AsyncEmbodiedSACFSDPPolicy",
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        reward: Union["EmbodiedRewardWorker"] = None,
        critic=None,
    ):
        self.cfg = cfg
        self.max_saved_checkpoints = cfg.runner.get("max_saved_checkpoints", 0)
        if (
            isinstance(self.max_saved_checkpoints, bool)
            or not isinstance(self.max_saved_checkpoints, int)
            or self.max_saved_checkpoints < 0
        ):
            raise ValueError(
                "runner.max_saved_checkpoints must be a nonnegative integer"
            )
        # Only this runner instance's successful saves are eligible for pruning.
        # Existing checkpoints, including a source SFT model, are never scanned.
        self._saved_checkpoints: dict[Path, int] = {}
        self._last_checkpoint_step: int | None = None
        self._last_checkpoint_path: Path | None = None
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        self.overlap_env_bootstrap = bool(
            self.cfg.runner.get("overlap_env_bootstrap", False)
        )
        for flag in ("eval_before_train", "log_eval_history"):
            if not isinstance(cfg.runner.get(flag, False), bool):
                raise ValueError(f"runner.{flag} must be boolean")
        if cfg.runner.get("eval_before_train", False) and (
            cfg.runner.get("val_check_interval", -1) <= 0
            or cfg.runner.get("use_training_pipeline", False)
        ):
            raise ValueError(
                "runner.eval_before_train requires enabled evaluation and "
                "synchronous training"
            )
        self.max_consecutive_zero_success = cfg.runner.get(
            "max_consecutive_zero_success", 0
        )
        if (
            isinstance(self.max_consecutive_zero_success, bool)
            or not isinstance(self.max_consecutive_zero_success, int)
            or self.max_consecutive_zero_success < 0
        ):
            raise ValueError(
                "runner.max_consecutive_zero_success must be a nonnegative integer"
            )
        algorithm = cfg.get("algorithm", {}) or {}
        guard = algorithm.get("ppo_update_guard", {}) or {}
        self._ppo_guard_enabled = guard.get("enabled", False)
        if not isinstance(self._ppo_guard_enabled, bool):
            raise ValueError("algorithm.ppo_update_guard.enabled must be a boolean")
        if self.max_consecutive_zero_success and not self._ppo_guard_enabled:
            raise ValueError(
                "runner.max_consecutive_zero_success requires "
                "algorithm.ppo_update_guard.enabled=true"
            )
        if self._ppo_guard_enabled and (
            self.overlap_env_bootstrap
            or cfg.runner.get("use_training_pipeline", False)
            or type(self).run is not EmbodiedRunner.run
        ):
            raise ValueError(
                "Guarded PPO stopping requires synchronous training without "
                "overlap_env_bootstrap or use_training_pipeline, using "
                "EmbodiedRunner.run"
            )
        self._consecutive_zero_success = 0

        # Step-gated profiling: ``cluster.profiling.steps`` lists the global step
        profiling_raw = self.cfg.cluster.get("profiling", None)
        profiling_enabled = profiling_raw is not None and bool(
            profiling_raw.get("enabled", True)
        )
        profile_steps_raw = (
            profiling_raw.get("steps", None) if profiling_enabled else None
        )
        self._profile_all_steps = profiling_enabled and profile_steps_raw is None
        self._profile_steps: set[int] | None = (
            {int(s) for s in profile_steps_raw}
            if profile_steps_raw is not None
            else None
        )

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.reward is not None:
            self.reward_channel = Channel.create("Reward")
        else:
            self.reward_channel = None

        # this timer checks if we should stop training
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

        # Async logging setup
        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    def _log_worker(self):
        """Background thread for processing log messages."""
        while not self.stop_logging:
            try:
                # Wait for log message with timeout
                log_func, args = self.log_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                log_func(*args)
            except Exception:
                self.logger.exception("Failed to print queued training metrics")
            finally:
                self.log_queue.task_done()

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        """Async version that puts table printing in queue."""
        self.log_queue.put(
            (
                print_metrics_table,
                (
                    step,
                    total_steps,
                    start_time,
                    metrics,
                    start_step,
                    self.metric_logger.log_path,
                ),
            )
        )

    def init_workers(self):
        """Initialize workers, optionally offloading the actor before simulators.

        ``runner.actor_first_initialization`` defaults to false. Enable it with
        actor offload to finish the actor's temporary GPU allocations before
        colocated environment workers create their persistent simulator state.
        Checkpoint restoration still happens after all workers initialize.
        """
        actor_first = self.cfg.runner.get("actor_first_initialization", False)
        if not isinstance(actor_first, bool):
            raise ValueError("runner.actor_first_initialization must be a boolean")
        if actor_first and (
            self.cfg.get("actor", {}).get("enable_offload", False) is not True
        ):
            raise ValueError(
                "runner.actor_first_initialization requires actor.enable_offload=true"
            )
        if actor_first:
            self.actor.init_worker().wait()

        # create worker in order to decrease the maximum memory usage
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        if self.reward is not None:
            self.reward.init_worker().wait()

        rollout_handle.wait()
        env_handle.wait()
        if not actor_first:
            self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        self.global_step = parse_global_step_from_checkpoint_path(resume_dir)
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        # Model/optimizer/RNG state lives in the actor checkpoint, while an
        # external simulator cannot be snapshotted portably.  Give environments
        # with a deterministic reconstruction contract (LBench) the completed
        # global-step cursor before any rollout bootstrap/reset occurs.
        self.env.restore_rollout_progress(self.global_step).wait()

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | None) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    @staticmethod
    def _split_pipeline_actor_results(
        results: list[dict] | None,
    ) -> tuple[list[dict], list[dict]]:
        if not results:
            return [], []
        rollout_metrics = [result.get("rollout_metrics", {}) for result in results]
        training_metrics = [result.get("training_metrics", {}) for result in results]
        return rollout_metrics, training_metrics

    def _maybe_eval_and_checkpoint(
        self, step: int, *, skip_evaluation: bool = False
    ) -> dict:
        run_val, save_model, _ = check_progress(
            self.global_step,
            self.max_steps,
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
            1.0,
            run_time_exceeded=False,
        )

        eval_metrics = {}
        # Preserve the completed update before restarting simulator pools or
        # running a potentially long evaluation. Evaluation uses these weights.
        if save_model:
            self._save_checkpoint()

        if run_val and not skip_evaluation:
            with self.timer("eval"):
                self.update_rollout_weights()
                eval_metrics = self.evaluate()
                eval_metrics = self._record_evaluation(eval_metrics)

        return eval_metrics

    def _record_evaluation(self, metrics: dict, *, baseline: bool = False) -> dict:
        """Log evaluation against completed PPO rounds, including baseline zero."""
        result = {f"eval/{key}": value for key, value in metrics.items()}
        self.metric_logger.log(data=result, step=self.global_step)
        if self.cfg.runner.get("log_eval_history", False):
            record = {
                "completed_rounds": self.global_step,
                "baseline": baseline,
                "timestamp": time.time(),
                "metrics": {
                    key: float(value) if math.isfinite(float(value)) else None
                    for key, value in result.items()
                },
            }
            path = Path(self.cfg.runner.logger.log_path) / "eval_history.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        return result

    def _log_step_metrics(
        self,
        step: int,
        start_time: float,
        start_step: int,
        env_handle: Handle,
        rollout_handle: Handle,
        actor_training_handle: Handle,
        reward_handle: Handle | None,
        actor_rollout_metrics: list[dict],
        actor_training_metrics: list[dict],
        eval_metrics: dict,
        env_results: list | None = None,
    ) -> None:
        time_metrics = self.timer.consume_durations()
        time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
        env_time_metrics, env_time_metrics_per_rank = env_handle.consume_durations(
            return_per_rank=True
        )
        rollout_time_metrics, rollout_time_metrics_per_rank = (
            rollout_handle.consume_durations(return_per_rank=True)
        )
        actor_time_metrics, actor_time_metrics_per_rank = (
            actor_training_handle.consume_durations(return_per_rank=True)
        )
        time_metrics.update({f"time/env/{k}": v for k, v in env_time_metrics.items()})
        time_metrics.update(
            {f"time/rollout/{k}": v for k, v in rollout_time_metrics.items()}
        )
        time_metrics.update(
            {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
        )
        if self.reward is not None:
            assert reward_handle is not None
            reward_time_metrics, reward_time_metrics_per_rank = (
                reward_handle.consume_durations(return_per_rank=True)
            )
            time_metrics.update(
                {f"time/reward/{k}": v for k, v in reward_time_metrics.items()}
            )

        if env_results is None:
            env_results = env_handle.wait()
        env_results_list = [results for results in env_results if results is not None]
        env_metrics = compute_evaluate_metrics(env_results_list)
        env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
        ranked_env_results = [
            {"rank": rank, "env": rank_metrics}
            for rank, rank_metrics in enumerate(env_results)
            if rank_metrics is not None
        ]
        _, env_metrics_per_rank = self._process_ranked_eval_results(
            ranked_env_results, metric_field="env"
        )

        rollout_metrics = {
            f"rollout/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_rollout_metrics).items()
        }
        training_metrics = {
            f"train/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_training_metrics).items()
        }

        self.metric_logger.log(env_metrics, step)
        self.metric_logger.log(rollout_metrics, step)
        self.metric_logger.log(time_metrics, step)
        self.metric_logger.log(training_metrics, step)
        self._log_ranked_metrics(
            metrics_list=actor_rollout_metrics,
            step=step,
            prefix="rollout",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_training_metrics,
            step=step,
            prefix="train",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_time_metrics_per_rank,
            step=step,
            prefix="time/actor",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=rollout_time_metrics_per_rank,
            step=step,
            prefix="time/rollout",
            worker_group_name=self.rollout.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_time_metrics_per_rank,
            step=step,
            prefix="time/env",
            worker_group_name=self.env.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_metrics_per_rank,
            step=step,
            prefix="env",
            worker_group_name=self.env.worker_group_name,
        )
        if self.reward is not None:
            self._log_ranked_metrics(
                metrics_list=reward_time_metrics_per_rank,
                step=step,
                prefix="time/reward",
                worker_group_name=self.reward.worker_group_name,
            )

        # Keep the dictionaries already handed to loggers immutable: evaluation
        # uses completed-round steps, not this training batch's zero-based step.
        logging_metrics = dict(time_metrics)
        logging_metrics.update(eval_metrics)
        logging_metrics.update(env_metrics)
        logging_metrics.update(rollout_metrics)
        logging_metrics.update(training_metrics)

        self.print_metrics_table_async(
            step, self.max_steps, start_time, logging_metrics, start_step
        )

    def _finish_run(self) -> None:
        # Keep the consumer alive until the final round's queued logs are drained.
        self.log_queue.join()
        self.stop_logging = True
        self.log_thread.join(timeout=1.0)
        self.metric_logger.finish()

    def _maybe_stop_guarded_training(
        self, actor_training_metrics: list[dict], step: int
    ) -> bool:
        """Save and stop after a completed synchronous round trips a PPO guard."""
        if not self._ppo_guard_enabled:
            return False

        success_counts = []
        zero_reward_flags = []
        abort_ranks = []
        abort_reasons = {}
        errors = []
        for rank, metrics in enumerate(actor_training_metrics):
            try:
                count = float(metrics["guard/positive_reward_count"])
                abort = float(metrics["guard/abort"])
                zero_reward = float(metrics["guard/zero_reward_batch"])
                if (
                    not math.isfinite(count)
                    or count < 0
                    or not count.is_integer()
                    or abort not in (0.0, 1.0)
                    or zero_reward not in (0.0, 1.0)
                ):
                    raise ValueError("invalid guard count or flag")
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                errors.append(f"actor rank {rank}: {error}")
                continue
            success_counts.append(int(count))
            zero_reward_flags.append(bool(zero_reward))
            if abort:
                abort_ranks.append(rank)
                abort_reasons[str(rank)] = [
                    key.removeprefix("guard/")
                    for key in (
                        "guard/invalid_rollout",
                        "guard/initial_probability_mismatch",
                        "guard/invalid_probability",
                        "guard/final_probability_mismatch",
                        "guard/nonfinite_gradient",
                    )
                    if metrics.get(key, 0)
                ]

        # The actor has already reduced this count globally. Summing per-rank
        # copies would overcount successes; disagreement instead fails closed.
        if not success_counts:
            errors.append("No valid global positive-reward count was returned")
        elif len(set(success_counts)) != 1 or len(set(zero_reward_flags)) != 1:
            errors.append("Actor ranks disagree on global reward metrics")
        success_count = success_counts[0] if success_counts and not errors else None
        if success_count is not None:
            self._consecutive_zero_success = (
                self._consecutive_zero_success + 1 if success_count == 0 else 0
            )

        reason = None
        if errors:
            reason = "invalid_guard_metrics"
        elif abort_ranks:
            reason = "actor_guard_abort"
        elif (
            self.max_consecutive_zero_success
            and self._consecutive_zero_success >= self.max_consecutive_zero_success
        ):
            reason = "max_consecutive_zero_success"
        self.metric_logger.log(
            {
                "train/guard/consecutive_zero_success": self._consecutive_zero_success,
                "train/guard/runner_stop": float(reason is not None),
            },
            step,
        )
        if reason is None:
            return False

        reused_checkpoint = self._last_checkpoint_step == self.global_step
        if not reused_checkpoint:
            self._save_checkpoint()
        assert self._last_checkpoint_path is not None
        details = {
            "reason": reason,
            "global_step": self.global_step,
            "completed_rounds": self.global_step,
            "metric_step": step,
            "configured_max_steps": self.max_steps,
            "positive_reward_count": success_count,
            "zero_reward_batch": zero_reward_flags[0] if not errors else None,
            "consecutive_zero_success": self._consecutive_zero_success,
            "max_consecutive_zero_success": self.max_consecutive_zero_success,
            "actor_abort_ranks": abort_ranks,
            "actor_abort_reasons": abort_reasons,
            "actor_abort_diagnostics": {
                str(rank): {
                    key.removeprefix("guard/abort_"): float(value)
                    if math.isfinite(float(value))
                    else None
                    for key, value in actor_training_metrics[rank].items()
                    if key.startswith("guard/abort_")
                }
                for rank in abort_ranks
            },
            "errors": errors,
            "checkpoint": str(self._last_checkpoint_path),
            "reused_checkpoint": reused_checkpoint,
            "automatic_resume": False,
        }
        self._write_stop_reason(details)
        self.logger.warning(
            "Stopped PPO after completed round %s: %s; positive rewards=%s, "
            "consecutive zero-success rounds=%s/%s; checkpoint=%s",
            self.global_step,
            reason,
            success_count,
            self._consecutive_zero_success,
            self.max_consecutive_zero_success,
            self._last_checkpoint_path,
        )
        return True

    def _write_stop_reason(self, details: dict) -> None:
        """Publish the stop record only after the complete checkpoint is saved."""
        output_dir = Path(self.cfg.runner.logger.log_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_dir,
                prefix=".stop_reason.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                json.dump(
                    details, output, indent=2, ensure_ascii=False, allow_nan=False
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, output_dir / "stop_reason.json")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _should_profile_step(self, step_idx: int) -> bool:
        return self._profile_all_steps or (
            self._profile_steps is not None and step_idx in self._profile_steps
        )

    def _open_profiling_window(self, step_idx: int) -> None:
        """Dispatch ``start_profile`` to all compute worker groups for this step."""
        self.logger.info(f"Opening profiling window at step {step_idx}")
        self.actor.start_profile(step_idx).wait()
        self.rollout.start_profile(step_idx).wait()
        self.env.start_profile(step_idx).wait()

    def _close_profiling_window(self, step_idx: int) -> None:
        """Dispatch ``stop_profile`` to all compute worker groups."""
        self.actor.stop_profile().wait()
        self.rollout.stop_profile().wait()
        self.env.stop_profile().wait()
        self.logger.info(f"Closed profiling window at step {step_idx}")

    def run(self):
        if self.cfg.runner.get("use_training_pipeline", False):
            return self.run_pipeline()

        if self.cfg.runner.get("eval_before_train", False):
            self.actor.set_global_step(self.global_step).wait()
            self.rollout.set_global_step(self.global_step).wait()
            self.update_rollout_weights()
            with self.timer("eval_before_train"):
                self._record_evaluation(self.evaluate(), baseline=self.global_step == 0)
            # Baseline/resume evaluation is outside the training step. Consume
            # it now so the first periodic/final evaluation can reuse the timer
            # and the first step's metrics do not include this startup cost.
            pretrain_time_metrics = self.timer.consume_durations()
            self.metric_logger.log(
                data={
                    f"time/{key}": value for key, value in pretrain_time_metrics.items()
                },
                step=self.global_step,
            )

        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step).wait()
            self.rollout.set_global_step(self.global_step).wait()

            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            with self.timer("step", trace_args={"step_idx": _step}):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                with self.timer("generate_rollouts"):
                    env_handle: Handle = self.env.interact(
                        input_channel=self.env_channel,
                        rollout_channel=self.rollout_channel,
                        reward_channel=self.reward_channel,
                        actor_channel=self.actor_channel,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    reward_handle = None
                    if self.reward is not None:
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.reward_channel,
                            output_channel=self.env_channel,
                        )
                    self.actor.recv_rollout_trajectories(
                        input_channel=self.actor_channel
                    ).wait()
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                # actor training.
                with self.timer("actor_training"):
                    actor_training_handle: Handle = self.actor.run_training()
                    env_bootstrap_handle: Handle | None = None
                    if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                        env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                            rollout_channel=self.rollout_channel
                        )

                    actor_training_metrics = actor_training_handle.wait()
                    if env_bootstrap_handle is not None:
                        env_bootstrap_handle.wait()

                self.global_step += 1
                # Evaluation may release and rebuild Isaac pools. Join the
                # completed interaction before requesting that phase switch.
                env_results = (
                    env_handle.wait()
                    if self.cfg.runner.val_check_interval > 0
                    else None
                )
                eval_metrics = self._maybe_eval_and_checkpoint(
                    _step,
                    skip_evaluation=any(
                        bool(metrics.get("guard/abort", False))
                        for metrics in actor_training_metrics
                    ),
                )

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            self._log_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=reward_handle,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
                env_results=env_results,
            )

            if self._maybe_stop_guarded_training(actor_training_metrics, _step):
                break

        self._finish_run()

    def run_pipeline(self):
        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step).wait()
            self.rollout.set_global_step(self.global_step).wait()

            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            with self.timer("step", trace_args={"step_idx": _step}):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                env_handle: Handle = self.env.interact(
                    input_channel=self.env_channel,
                    rollout_channel=self.rollout_channel,
                    reward_channel=self.reward_channel,
                    actor_channel=self.actor_channel,
                )
                rollout_handle: Handle = self.rollout.generate(
                    input_channel=self.rollout_channel,
                    output_channel=self.env_channel,
                )
                reward_handle = None
                if self.reward is not None:
                    reward_handle: Handle = self.reward.compute_rewards(
                        input_channel=self.reward_channel,
                        output_channel=self.env_channel,
                    )
                # actor training.
                actor_training_handle: Handle = self.actor.run_training(
                    input_channel=self.actor_channel
                )
                with self.timer("generate_rollouts"):
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                env_bootstrap_handle: Handle | None = None
                if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                    env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    )

                actor_results = actor_training_handle.wait()
                actor_rollout_metrics, actor_training_metrics = (
                    self._split_pipeline_actor_results(actor_results)
                )
                if env_bootstrap_handle is not None:
                    env_bootstrap_handle.wait()

                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(_step)

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            self._log_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=reward_handle,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        self._finish_run()

    def _save_checkpoint(self):
        # Invalidate before attempting a rewrite; a failed save cannot be reused.
        self._last_checkpoint_step = None
        self._last_checkpoint_path = None
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        checkpoint_dir = Path(base_output_dir).absolute()
        if self.max_saved_checkpoints:
            for path in (
                checkpoint_dir.parent,
                checkpoint_dir,
                checkpoint_dir / "actor",
            ):
                if path.is_symlink():
                    raise ValueError(
                        f"Refusing to save through checkpoint symlink: {path}"
                    )
            # A failed rewrite must not retain its previous completion status.
            self._saved_checkpoints.pop(checkpoint_dir, None)
        os.makedirs(actor_save_path, exist_ok=True)
        if self.max_saved_checkpoints:
            (checkpoint_dir / _CHECKPOINT_COMPLETE_MARKER).unlink(missing_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()
        if self.max_saved_checkpoints:
            self._mark_checkpoint_complete(checkpoint_dir)
            self._saved_checkpoints[checkpoint_dir] = self.global_step
            self._prune_checkpoints(checkpoint_dir)
        self._last_checkpoint_step = self.global_step
        self._last_checkpoint_path = checkpoint_dir

    def _mark_checkpoint_complete(self, checkpoint_dir: Path) -> None:
        """Publish completion atomically after every actor rank finishes saving."""
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=checkpoint_dir,
                prefix=f"{_CHECKPOINT_COMPLETE_MARKER}.",
                suffix=".tmp",
                delete=False,
            ) as marker:
                temporary_path = Path(marker.name)
                marker.write(f"{self.global_step}\n")
                marker.flush()
                os.fsync(marker.fileno())
            os.replace(temporary_path, checkpoint_dir / _CHECKPOINT_COMPLETE_MARKER)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _prune_checkpoints(self, current_checkpoint: Path) -> None:
        """Prune older complete saves registered by this runner, when opted in."""
        if not self.max_saved_checkpoints:
            return
        checkpoint_root = current_checkpoint.parent
        if checkpoint_root.is_symlink():
            self.logger.warning(
                "Skipping checkpoint retention for symlink %s", checkpoint_root
            )
            return

        completed = []
        for checkpoint_dir, step in self._saved_checkpoints.items():
            if (
                checkpoint_dir.parent != checkpoint_root
                or checkpoint_dir.name != f"global_step_{step}"
                or checkpoint_dir.is_symlink()
                or not checkpoint_dir.is_dir()
            ):
                continue
            marker = checkpoint_dir / _CHECKPOINT_COMPLETE_MARKER
            if marker.is_symlink() or not marker.is_file():
                continue
            completed.append((step, checkpoint_dir))

        completed.sort(key=lambda item: item[0], reverse=True)
        for step, checkpoint_dir in completed[self.max_saved_checkpoints :]:
            if checkpoint_dir == current_checkpoint or step >= self.global_step:
                continue
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError:
                self.logger.warning(
                    "Failed to prune checkpoint %s", checkpoint_dir, exc_info=True
                )
            else:
                self._saved_checkpoints.pop(checkpoint_dir)
                self.logger.info("Pruned checkpoint %s", checkpoint_dir)

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
