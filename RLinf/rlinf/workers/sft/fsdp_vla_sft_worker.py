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
import inspect
import json
import logging
import os
import pathlib
import shutil
import tempfile
from collections.abc import Iterator
from typing import Any

import torch
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.data.datasets.common.epoch import forward_set_epoch
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


def _sft_forward_kwargs(model_type: SupportedModel) -> dict[str, bool]:
    """Return model-specific arguments for the offline VLA SFT forward."""
    if model_type == SupportedModel.OPENPI:
        # Supervise every predicted step, even when inference executes a
        # shorter chunk. Only exclude the padded action channels, matching
        # LeRobot's offline SFT loss. DAgger keeps its execution-chunk mask.
        return {"use_action_dim_loss": True}
    return {}


class _EpochBoundedOpenPIDataLoader:
    """Expose one finite epoch from OpenPI's otherwise infinite loader.

    RLinf's SFT worker advances the distributed sampler only after an iterator
    raises ``StopIteration``. Official OpenPI's ``TorchDataLoader`` loops over
    its inner PyTorch loader forever, so without this adapter every epoch uses
    sampler epoch zero and a checkpoint cannot identify an in-epoch position.
    """

    def __init__(self, data_loader: Any):
        self._official_loader = data_loader
        self._data_loader = getattr(data_loader, "_data_loader", None)
        if self._data_loader is None:
            raise TypeError("OpenPI data loader does not expose its inner loader")
        self._torch_loader = getattr(self._data_loader, "_data_loader", None)
        if self._torch_loader is None:
            self._torch_loader = getattr(self._data_loader, "torch_loader", None)
        if self._torch_loader is None:
            raise TypeError("OpenPI data loader does not expose a PyTorch DataLoader")
        # Exact mid-epoch resume replays batches from the generator state saved
        # at the beginning of this epoch. Persistent workers retain their own
        # advanced Python/NumPy RNG states across epochs, which a fresh process
        # cannot reconstruct from that generator state. Recreate workers at each
        # epoch boundary so replay is deterministic at the cost of a small,
        # infrequent worker-startup overhead.
        if getattr(self._torch_loader, "persistent_workers", False):
            self._torch_loader = self._rebuild_without_persistent_workers(
                self._torch_loader
            )
            # OpenPI's TorchDataLoader.__iter__ reads this private field on
            # every new iterator. Replace it before the first iterator starts;
            # mutating torch DataLoader.persistent_workers after construction
            # is explicitly rejected by PyTorch.
            self._data_loader._data_loader = self._torch_loader
        self._num_batches = len(self._torch_loader)
        if self._num_batches <= 0:
            raise ValueError(
                "The local LBench dataset shard has no full micro-batch; reduce "
                "actor.micro_batch_size or use more samples."
            )
        self._epoch_generator_state: torch.Tensor | None = None

    @staticmethod
    def _rebuild_without_persistent_workers(
        loader: torch.utils.data.DataLoader,
    ) -> torch.utils.data.DataLoader:
        """Clone OpenPI's standard torch loader with finite worker lifetimes."""

        if loader.batch_size is None:
            raise TypeError(
                "LBench exact-resume requires OpenPI's standard batched DataLoader"
            )
        kwargs: dict[str, Any] = {
            "dataset": loader.dataset,
            "batch_size": loader.batch_size,
            "sampler": loader.sampler,
            "num_workers": loader.num_workers,
            "collate_fn": loader.collate_fn,
            "pin_memory": loader.pin_memory,
            "drop_last": loader.drop_last,
            "timeout": loader.timeout,
            "worker_init_fn": loader.worker_init_fn,
            "generator": loader.generator,
            "persistent_workers": False,
        }
        parameters = inspect.signature(torch.utils.data.DataLoader).parameters
        optional_values = {
            "multiprocessing_context": getattr(loader, "multiprocessing_context", None),
            "prefetch_factor": getattr(loader, "prefetch_factor", None),
            "pin_memory_device": getattr(loader, "pin_memory_device", ""),
            "in_order": getattr(loader, "in_order", True),
        }
        for name, value in optional_values.items():
            if name not in parameters:
                continue
            if name in {"multiprocessing_context", "prefetch_factor"} and (
                loader.num_workers == 0 or value is None
            ):
                continue
            kwargs[name] = value
        return torch.utils.data.DataLoader(**kwargs)

    @property
    def sampler(self) -> Any:
        return getattr(self._torch_loader, "sampler", None)

    @property
    def dataset(self) -> Any:
        return getattr(self._torch_loader, "dataset", None)

    def data_config(self) -> Any:
        return self._official_loader.data_config()

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[Any]:
        generator = getattr(self._torch_loader, "generator", None)
        if generator is not None:
            self._epoch_generator_state = generator.get_state().clone()
        iterator = iter(self._official_loader)
        for batch_index in range(self._num_batches):
            try:
                yield next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "OpenPI data loader ended before its advertised epoch length "
                    f"({batch_index}/{self._num_batches} batches)"
                ) from exc

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_batches": self._num_batches,
            "epoch_generator_state": self._epoch_generator_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_batches = int(state.get("num_batches", -1))
        if saved_batches != self._num_batches:
            raise ValueError(
                "LBench data loader length changed across resume: "
                f"checkpoint={saved_batches}, current={self._num_batches}"
            )
        generator_state = state.get("epoch_generator_state")
        generator = getattr(self._torch_loader, "generator", None)
        if generator_state is not None:
            if generator is None:
                raise ValueError(
                    "Checkpoint has an OpenPI generator state but the current "
                    "PyTorch DataLoader has no generator"
                )
            generator.set_state(generator_state)


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI_RLINF:
            from rlinf.data.datasets.openpi_rlinf import (
                build_openpi_rlinf_sft_dataloader,
            )

            return build_openpi_rlinf_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.OPENPI:
            from rlinf.data.datasets.openpi_rlinf import (
                build_official_openpi_sft_dataloader,
            )

            data_loader, data_config = build_official_openpi_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
            if str(self.cfg.actor.model.openpi.config_name) == "pi05_lbench_joint":
                # The official OpenPI loader intentionally repeats forever.
                # LBench SFT needs a finite iterator so RLinf can advance the
                # DistributedSampler epoch and persist an exact resume position.
                data_loader = _EpochBoundedOpenPIDataLoader(data_loader)
            return data_loader, data_config
        elif model_type == SupportedModel.LINGBOTVLA:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.DREAMZERO:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.EVO1:
            from rlinf.models.embodiment.evo1.sft_builder import (
                build_evo1_sft_dataloader,
            )

            return build_evo1_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        with self.amp_context:
            output = self.model(
                forward_type=ForwardType.SFT,
                data=batch,
                **_sft_forward_kwargs(model_type),
            )

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def _export_openpi_assets(self, save_path: str, norm_stats_path: str) -> None:
        """Bundle the exact SFT normalization assets into a checkpoint."""
        source = pathlib.Path(str(norm_stats_path)).expanduser()
        if source.is_dir():
            source = source / "norm_stats.json"
        if not source.is_file():
            raise FileNotFoundError(
                f"Configured OpenPI norm_stats_path does not exist: {source}"
            )
        runtime_asset_id = getattr(self.data_config, "asset_id", None)
        if not runtime_asset_id:
            raise ValueError("OpenPI data_config.asset_id is required for export")

        # Also write the canonical asset id used when openpi_data is not
        # overridden. This makes both <global_step> and <global_step>/actor
        # directly deployable, with or without an explicit stats path.
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        config_name = str(self.cfg.actor.model.openpi.config_name)
        canonical_asset_id = get_openpi_config(config_name).data.repo_id
        asset_ids = list(
            dict.fromkeys([str(runtime_asset_id), str(canonical_asset_id)])
        )
        invalid_asset_ids = [
            asset_id
            for asset_id in asset_ids
            if not asset_id
            or pathlib.Path(asset_id).is_absolute()
            or ".." in pathlib.Path(asset_id).parts
        ]
        if invalid_asset_ids:
            raise ValueError(
                "OpenPI asset ids must be safe checkpoint-relative paths: "
                f"{invalid_asset_ids}"
            )
        actor_root = pathlib.Path(save_path)
        checkpoint_roots = [actor_root]
        if actor_root.name == "actor":
            checkpoint_roots.append(actor_root.parent)
        destinations = [
            str(pathlib.Path(asset_id) / "norm_stats.json") for asset_id in asset_ids
        ]
        for checkpoint_root in checkpoint_roots:
            for asset_id in asset_ids:
                destination = checkpoint_root / asset_id / "norm_stats.json"
                self._atomic_copy_file(source, destination)
        manifest = {
            "asset_ids": asset_ids,
            "config_name": config_name,
            "norm_stats": destinations,
            "action_horizon": int(self.cfg.actor.model.num_action_chunks),
            "action_dim": int(self.cfg.actor.model.action_dim),
            "compatibility": {
                "parameter_precision": str(
                    self.cfg.actor.model.openpi.get("parameter_precision", "openpi")
                ),
                "lerobot_compatible": bool(
                    self.cfg.actor.model.get("openpi_data", {}).get(
                        "lerobot_compatible", False
                    )
                ),
            },
        }
        for checkpoint_root in checkpoint_roots:
            self._atomic_json_dump(manifest, checkpoint_root / "openpi_assets.json")

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        # A rank-zero filesystem exception must be broadcast before any rank
        # continues, otherwise the remaining FSDP ranks can wait forever at a
        # later collective.
        norm_stats_path = self.cfg.actor.model.get("openpi_data", {}).get(
            "norm_stats_path"
        )
        asset_error: Exception | None = None
        # Standard OpenPI models already exported their effective assets in the
        # model manager. Keep this fallback for older model implementations.
        if (
            self._rank == 0
            and norm_stats_path
            and getattr(self, "_openpi_checkpoint_assets", None) is None
        ):
            try:
                self._export_openpi_assets(save_path, str(norm_stats_path))
            except Exception as exc:  # propagate the same failure to every rank
                asset_error = exc
        self._raise_rank_zero_io_error(asset_error, "exporting OpenPI assets")

        if isinstance(
            self.data_loader, (StatefulDataLoader, _EpochBoundedOpenPIDataLoader)
        ):
            state: dict[str, Any] | None = None
            rng_state: Any | None = None
            serialization_error: Exception | None = None
            try:
                state = {
                    "loader": self.data_loader.state_dict(),
                    "data_epoch": int(self._data_epoch),
                    "data_iter_offset": int(self._data_iter_offset),
                }
                rng_state = get_rng_state()
            except Exception as exc:
                serialization_error = exc
            # Every rank reaches this error collective before any rank enters
            # the object gathers below. Otherwise one local state_dict failure
            # would leave its peers blocked forever in all_gather_object.
            self._raise_any_rank_error(
                serialization_error, "serializing SFT data-loader and RNG state"
            )
            if state is None or rng_state is None:
                raise RuntimeError(
                    "SFT checkpoint state serialization completed without a value"
                )
            all_states = self._gather_rank_objects(state)
            all_rng_states = self._gather_rank_objects(rng_state)
            state_error: Exception | None = None
            if self._rank == 0:
                try:
                    self._atomic_torch_save(
                        all_states, os.path.join(save_path, "data.pt")
                    )
                    self._atomic_torch_save(
                        all_rng_states, os.path.join(save_path, "rng.pt")
                    )
                except Exception as exc:  # propagate before another collective
                    state_error = exc
            self._raise_rank_zero_io_error(
                state_error, "saving SFT data-loader and RNG state"
            )

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)
        # Safe fallback for checkpoints created before the dedicated per-rank
        # rng.pt file existed. Data-loader replay below may consume RNG again.
        model_checkpoint_rng_state = get_rng_state()

        if isinstance(
            self.data_loader, (StatefulDataLoader, _EpochBoundedOpenPIDataLoader)
        ):
            data_path = os.path.join(load_path, "data.pt")
            data_exists, all_states = self._load_rank_zero_object(
                data_path, "loading SFT data-loader state"
            )
            if not data_exists:
                if self._rank == 0:
                    logging.warning(
                        "Checkpoint %s has no data.pt; resuming model/optimizer but "
                        "starting the SFT loader at the beginning of its current "
                        "epoch.",
                        load_path,
                    )
                return
            rng_path = os.path.join(load_path, "rng.pt")
            rng_exists, all_rng_states = self._load_rank_zero_object(
                rng_path, "loading SFT RNG state"
            )

            restore_error: Exception | None = None
            try:
                if len(all_states) != self._world_size:
                    raise ValueError(
                        "SFT checkpoint world size differs from the current run: "
                        f"checkpoint={len(all_states)}, current={self._world_size}"
                    )
                state = all_states[self._rank]
                if "loader" in state:
                    self.data_loader.load_state_dict(state["loader"])
                    self._data_epoch = int(state.get("data_epoch", 0))
                    self._data_iter_offset = int(state.get("data_iter_offset", 0))
                else:
                    # Backward compatibility with the previous StatefulDataLoader
                    # checkpoint, which stored only loader.state_dict().
                    self.data_loader.load_state_dict(state)
                    self._data_epoch = 0
                    self._data_iter_offset = 0
                if self._data_epoch:
                    forward_set_epoch(self.data_loader, self._data_epoch)
                self.data_iter = iter(self.data_loader)
                if isinstance(self.data_loader, _EpochBoundedOpenPIDataLoader):
                    if self._data_iter_offset > len(self.data_loader):
                        raise ValueError(
                            "SFT checkpoint data offset exceeds the current epoch: "
                            f"{self._data_iter_offset} > {len(self.data_loader)}"
                        )
                    for _ in range(self._data_iter_offset):
                        next(self.data_iter)

                if rng_exists:
                    if len(all_rng_states) != self._world_size:
                        raise ValueError(
                            "SFT RNG checkpoint world size differs from the current "
                            f"run: checkpoint={len(all_rng_states)}, "
                            f"current={self._world_size}"
                        )
                    # Restore after replaying skipped batches, since DataLoader
                    # iterator construction may itself consume RNG state.
                    set_rng_state(all_rng_states[self._rank])
                else:
                    set_rng_state(model_checkpoint_rng_state)
                    if self._rank == 0:
                        logging.warning(
                            "Checkpoint %s has no rng.pt; restored the actor "
                            "checkpoint RNG after replaying the data offset.",
                            load_path,
                        )
            except Exception as exc:
                restore_error = exc
            self._raise_any_rank_error(
                restore_error, "restoring SFT data-loader and RNG state"
            )

    @staticmethod
    def _atomic_torch_save(value: Any, path: str) -> None:
        """Publish one torch checkpoint object with an atomic rename."""

        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
        )
        os.close(fd)
        try:
            torch.save(value, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _atomic_copy_file(source: pathlib.Path, destination: pathlib.Path) -> None:
        """Copy one checkpoint asset without exposing a partial destination."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        os.close(fd)
        try:
            shutil.copy2(source, temporary_path)
            os.replace(temporary_path, destination)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _atomic_json_dump(value: Any, destination: pathlib.Path) -> None:
        """Write one JSON checkpoint manifest with an atomic rename."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _load_rank_zero_object(
        self, path: str, context: str
    ) -> tuple[bool, Any | None]:
        """Read a shared checkpoint once and broadcast its value or error."""

        exists = False
        value = None
        error: Exception | None = None
        if self._rank == 0:
            exists = os.path.exists(path)
            if exists:
                try:
                    value = torch.load(path, map_location="cpu", weights_only=False)
                except Exception as exc:
                    error = exc

        message = None if error is None else f"{type(error).__name__}: {error}"
        payload = [exists, value, message]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(payload, src=0)
        if payload[2] is not None:
            failure = RuntimeError(f"{context} failed on rank 0: {payload[2]}")
            if self._rank == 0 and error is not None:
                raise failure from error
            raise failure
        return bool(payload[0]), payload[1]

    def _raise_any_rank_error(self, error: Exception | None, context: str) -> None:
        """Make any rank-local failure visible to every FSDP rank."""

        message = None if error is None else f"{type(error).__name__}: {error}"
        if torch.distributed.is_initialized():
            messages = [None] * self._world_size
            torch.distributed.all_gather_object(messages, message)
        else:
            messages = [message]
        failures = [
            f"rank {rank}: {value}"
            for rank, value in enumerate(messages)
            if value is not None
        ]
        if failures:
            failure = RuntimeError(f"{context} failed ({'; '.join(failures)})")
            if error is not None:
                raise failure from error
            raise failure

    def _gather_rank_objects(self, value: Any) -> list[Any]:
        if not torch.distributed.is_initialized():
            return [value]
        result = [None] * self._world_size
        torch.distributed.all_gather_object(result, value)
        return result

    def _raise_rank_zero_io_error(self, error: Exception | None, context: str) -> None:
        message = None
        if self._rank == 0 and error is not None:
            message = f"{type(error).__name__}: {error}"
        payload = [message]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(payload, src=0)
        if payload[0] is not None:
            failure = RuntimeError(f"{context} failed on rank 0: {payload[0]}")
            if self._rank == 0 and error is not None:
                raise failure from error
            raise failure

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type in (SupportedModel.OPENPI_RLINF, SupportedModel.OPENPI):
            if model_type == SupportedModel.OPENPI_RLINF:
                from rlinf.data.datasets.openpi_rlinf import (
                    get_official_openpi_sft_num_batches,
                    is_official_openpi_sft_dataloader,
                )

                num_batches = (
                    get_official_openpi_sft_num_batches(self.data_loader)
                    if is_official_openpi_sft_dataloader(self.data_loader)
                    else len(self.data_loader)
                )
            else:
                from rlinf.data.datasets.openpi_rlinf import (
                    get_official_openpi_sft_num_batches,
                )

                num_batches = get_official_openpi_sft_num_batches(self.data_loader)
        else:
            return super().get_max_steps_per_epoch()
        return max(1, num_batches // self.gradient_accumulation)
