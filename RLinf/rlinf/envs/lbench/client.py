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

"""Thin loader for LBench's transport-neutral external worker client."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from DictConfig, mapping, or a simple namespace."""

    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def to_plain(value: Any) -> Any:
    """Recursively remove config container types before crossing environments."""

    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) or (
        hasattr(value, "__iter__") and type(value).__name__ == "ListConfig"
    ):
        return [to_plain(item) for item in value]
    return value


def resolve_lbench_root(cfg: Any) -> Path:
    value = os.path.expandvars(
        str(cfg_get(cfg, "lbench_root", os.environ.get("LBENCH_ROOT", "")))
    )
    root = Path(value).expanduser().resolve() if value else None
    if root is None or not (root / "simbox_eval").is_dir():
        raise FileNotFoundError(
            "LBench repository not found; set env.*.lbench_root or LBENCH_ROOT"
        )
    return root


def _validate_lbench_module_origins(root: Path, module_names: tuple[str, ...]) -> None:
    """Reject modules cached from a different LBench checkout.

    Merely prepending ``root`` to ``sys.path`` cannot replace an already
    imported package.  Continuing in that situation could make the RLinf
    driver and external Isaac workers use different protocol implementations.
    """

    expected_root = root.resolve()
    mismatches: dict[str, list[str]] = {}
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is None:
            mismatches[module_name] = ["<not imported>"]
            continue
        candidates: list[Path] = []
        module_file = getattr(module, "__file__", None)
        if module_file:
            candidates.append(Path(module_file).resolve())
        for package_path in getattr(module, "__path__", ()):
            candidates.append(Path(package_path).resolve())
        if not candidates or not any(
            candidate == expected_root or expected_root in candidate.parents
            for candidate in candidates
        ):
            mismatches[module_name] = [str(candidate) for candidate in candidates] or [
                "<unknown origin>"
            ]
    if mismatches:
        raise ImportError(
            "simbox_eval is already loaded from a different LBench checkout; "
            f"expected modules under {expected_root}, got {mismatches}. "
            "Restart the RLinf/Ray process after setting LBENCH_ROOT."
        )


def import_lbench_modules(cfg: Any):
    """Import only LBench's pure client/protocol modules in the RLinf process."""

    root = resolve_lbench_root(cfg)
    root_text = str(root)
    # Always give the configured checkout precedence for the first import.
    # If another checkout was imported earlier, the origin check below fails
    # explicitly instead of silently mixing parent/worker protocol versions.
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    from simbox_eval.action_mapping import map_joint_action_8d
    from simbox_eval.parallel.external_worker import ExternalProcessWorker
    from simbox_eval.parallel.protocol import EpisodeSpec
    from simbox_eval.parallel.vector_env import SimBoxSubprocVectorEnv

    _validate_lbench_module_origins(
        root,
        (
            "simbox_eval",
            "simbox_eval.action_mapping",
            "simbox_eval.parallel.external_worker",
            "simbox_eval.parallel.protocol",
            "simbox_eval.parallel.vector_env",
        ),
    )

    return (
        root,
        map_joint_action_8d,
        ExternalProcessWorker,
        EpisodeSpec,
        SimBoxSubprocVectorEnv,
    )


def create_vector_client(cfg: Any, num_envs: int, worker_id_base: int):
    """Create exactly ``num_envs`` external Isaac workers."""

    (
        root,
        _map_action,
        external_worker_cls,
        _episode_spec,
        vector_env_cls,
    ) = import_lbench_modules(cfg)
    python_value = os.path.expandvars(
        str(cfg_get(cfg, "isaac_python", os.environ.get("LBENCH_ISAAC_PYTHON", "")))
    )
    if not python_value:
        raise ValueError(
            "Isaac Python is required; set env.*.isaac_python or LBENCH_ISAAC_PYTHON"
        )
    # Preserve venv/Conda launcher symlinks. Resolving ``env/bin/python`` can
    # turn it into the base interpreter and silently lose the selected Isaac
    # environment before ExternalProcessWorker has a chance to launch it.
    python_executable = os.path.abspath(str(Path(python_value).expanduser()))
    backend_factory_path = str(
        cfg_get(
            cfg,
            "backend_factory_path",
            "simbox_eval.parallel.simbox_backend:create_simbox_backend",
        )
    )
    connect_timeout_s = float(cfg_get(cfg, "connect_timeout_s", 120.0))
    isaac_devices = cfg_get(cfg, "isaac_cuda_visible_devices", None)
    if isaac_devices is None:
        isaac_devices = os.environ.get("LBENCH_ISAAC_GPUS")
    if isinstance(isaac_devices, str):
        isaac_devices = [
            item.strip() for item in isaac_devices.split(",") if item.strip()
        ]
    elif isaac_devices is not None:
        isaac_devices = [str(item) for item in isaac_devices]
    if isaac_devices is not None and not isaac_devices:
        raise ValueError("isaac_cuda_visible_devices must not be empty")
    if isaac_devices is not None and len(set(isaac_devices)) != len(isaac_devices):
        raise ValueError("isaac_cuda_visible_devices must not contain duplicates")
    assignment = cfg_get(cfg, "isaac_gpu_assignment", "round_robin")
    if assignment not in {"round_robin", "env_rank"}:
        raise ValueError(f"Unknown isaac_gpu_assignment: {assignment!r}")
    if assignment == "env_rank":
        configured_total = cfg_get(cfg, "total_num_envs")
        if (
            not isaac_devices
            or num_envs < 1
            or configured_total != num_envs * len(isaac_devices)
            or worker_id_base < 0
            or worker_id_base % num_envs
            or worker_id_base // num_envs >= len(isaac_devices)
        ):
            raise ValueError(
                "env_rank GPU assignment requires one device per equally sized "
                "environment shard and an aligned worker_id_base"
            )
    isaac_ld_preload = os.path.expandvars(
        str(
            cfg_get(
                cfg,
                "isaac_ld_preload",
                os.environ.get("LBENCH_ISAAC_LD_PRELOAD", ""),
            )
        )
    ).strip()

    def worker_factory(_worker_id: int, env_id: int):
        child_env = {}
        if isaac_devices is not None:
            device_index = (
                worker_id_base // num_envs
                if assignment == "env_rank"
                else (worker_id_base + env_id) % len(isaac_devices)
            )
            child_env["CUDA_VISIBLE_DEVICES"] = isaac_devices[device_index]
        if isaac_ld_preload:
            # Keep the workaround scoped to the independently launched Isaac
            # process.  Preloading this library into RLinf's model workers can
            # introduce unrelated OpenMP/runtime conflicts.
            child_env["LD_PRELOAD"] = isaac_ld_preload
        return external_worker_cls(
            worker_id=worker_id_base + env_id,
            env_id=env_id,
            python_executable=python_executable,
            repository_root=str(root),
            backend_factory_path=backend_factory_path,
            connect_timeout_s=connect_timeout_s,
            child_env_overrides=child_env or None,
        )

    backend_config = to_plain(cfg_get(cfg, "backend_config", {}))
    if not isinstance(backend_config, dict):
        raise TypeError("env.*.backend_config must be a mapping")
    startup_options = {}
    startup_batch_size = cfg_get(cfg, "startup_batch_size")
    if startup_batch_size is not None:
        startup_options["startup_batch_size"] = startup_batch_size
    return vector_env_cls(
        int(num_envs),
        worker_factory=worker_factory,
        init_payload={"backend_config": backend_config},
        startup_timeout_s=float(cfg_get(cfg, "startup_timeout_s", 180.0)),
        request_timeout_s=float(cfg_get(cfg, "request_timeout_s", 120.0)),
        **startup_options,
    )


__all__ = [
    "cfg_get",
    "create_vector_client",
    "import_lbench_modules",
    "resolve_lbench_root",
    "to_plain",
]
