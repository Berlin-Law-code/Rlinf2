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

"""Keep the effective OpenPI preprocessing contract with every checkpoint."""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch.distributed as dist


def read_checkpoint_manifest(checkpoint_dir: str | Path) -> dict:
    """Read assets without opening model weights; accept root and actor entries."""
    root = Path(checkpoint_dir)
    for path in (root / "openpi_assets.json", root / "actor/openpi_assets.json"):
        if path.is_file():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError(f"OpenPI asset manifest must be an object: {path}")
            return manifest
    return {}


def _relative_asset(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"OpenPI assets must use checkpoint-relative paths: {value!r}")
    return path


def _stats_file(path: str | Path) -> Path:
    result = Path(path).expanduser()
    if result.is_dir():
        result /= "norm_stats.json"
    if not result.is_file():
        raise FileNotFoundError(f"OpenPI normalization file does not exist: {result}")
    if result.name != "norm_stats.json":
        raise ValueError(f"OpenPI normalization file must be norm_stats.json: {result}")
    return result.resolve()


def _load_stats(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"OpenPI normalization must be an object: {path}")
    payload = payload.get("norm_stats", payload)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            f"OpenPI normalization must contain feature statistics: {path}"
        )
    return payload


def validate_matching_norm_stats(expected_path: Path, actual_path: Path) -> None:
    """Reject any changed normalization value, accepting JSON formatting changes."""
    expected, actual = _load_stats(expected_path), _load_stats(actual_path)
    if expected.keys() != actual.keys():
        raise ValueError("OpenPI normalization differs from checkpoint feature keys")
    for feature, values in expected.items():
        other = actual[feature]
        if not isinstance(values, dict) or not isinstance(other, dict):
            raise ValueError(f"Invalid OpenPI normalization feature {feature!r}")
        if values.keys() != other.keys():
            raise ValueError(f"OpenPI normalization differs from checkpoint: {feature}")
        for field, value in values.items():
            if value is None and other[field] is None:
                continue
            left, right = np.asarray(value), np.asarray(other[field])
            if (
                not np.issubdtype(left.dtype, np.number)
                or not np.issubdtype(right.dtype, np.number)
                or not np.isfinite(left).all()
                or not np.isfinite(right).all()
                or not np.array_equal(left, right)
            ):
                raise ValueError(
                    "OpenPI normalization differs from checkpoint: "
                    f"{feature}.{field} ({actual_path} != {expected_path})"
                )


def resolve_checkpoint_norm_stats(
    checkpoint_dir: str | Path,
    explicit_path: str | Path | None,
    *,
    canonical_asset_id: str | None = None,
    allow_explicit_override: bool = False,
) -> Path | None:
    """Prefer bundled statistics and fail if an explicit override changes them.

    Base models and legacy checkpoints without bundled statistics may still use
    an external asset. A manifest naming missing assets is an incomplete
    checkpoint, and must not silently fall back to an unrelated external file.
    A new fine-tuning run may explicitly select new dataset statistics without
    modifying the source checkpoint. Resume and inference keep the default.
    """
    if not isinstance(allow_explicit_override, bool):
        raise ValueError("allow_explicit_override must be a boolean")
    if allow_explicit_override and explicit_path is None:
        raise ValueError("Normalization override requires an explicit norm_stats_path")
    root = Path(checkpoint_dir)
    manifest = read_checkpoint_manifest(root)
    declared = manifest.get("norm_stats", [])
    if not isinstance(declared, list):
        raise ValueError("OpenPI manifest norm_stats must be a list")
    bundled = []
    for name in declared:
        relative = _relative_asset(name)
        candidates = (root / relative, root / "actor" / relative)
        match = next((path for path in candidates if path.is_file()), None)
        if match is None:
            raise FileNotFoundError(
                f"Checkpoint normalization asset is missing: {name}"
            )
        bundled.append(_stats_file(match))
    if (
        not bundled
        and canonical_asset_id
        and not Path(canonical_asset_id).is_absolute()
        and ".." not in Path(canonical_asset_id).parts
        and Path(canonical_asset_id) != Path(".")
    ):
        relative = _relative_asset(canonical_asset_id) / "norm_stats.json"
        for candidate in (root / relative, root / "actor" / relative):
            if candidate.is_file():
                bundled.append(_stats_file(candidate))
                break
    explicit = _stats_file(explicit_path) if explicit_path is not None else None
    if not bundled:
        return explicit
    selected = bundled[0]
    for candidate in bundled[1:]:
        validate_matching_norm_stats(selected, candidate)
    if explicit is not None:
        if allow_explicit_override:
            validate_matching_norm_stats(explicit, explicit)
        try:
            validate_matching_norm_stats(selected, explicit)
        except ValueError:
            if not allow_explicit_override:
                raise
            logging.getLogger(__name__).warning(
                "Initializing fine-tuning with external normalization %s instead "
                "of checkpoint normalization %s; source checkpoint is unchanged.",
                explicit,
                selected,
            )
            return explicit
    return selected


def build_checkpoint_assets(
    *,
    config_name: str,
    norm_stats: dict,
    asset_ids: list[str],
    parameter_precision: str,
    lerobot_compatible: bool,
    action_horizon: int,
    execute_horizon: int,
    action_dim: int,
) -> dict:
    """Snapshot actual transforms/config, independent of mutable source files."""
    # Local dataset paths are repo identifiers, not checkpoint-relative assets.
    # Legacy non-LBench configurations can use them; give those a safe alias.
    asset_ids = list(
        dict.fromkeys(
            str(value)
            for value in asset_ids
            if value
            and not Path(value).is_absolute()
            and ".." not in Path(value).parts
            and Path(value) != Path(".")
        )
    ) or ["normalization"]
    for asset_id in asset_ids:
        _relative_asset(asset_id)
    if not 0 < execute_horizon <= action_horizon or action_dim <= 0:
        raise ValueError(
            "Invalid OpenPI prediction/execution horizon or action dimension"
        )
    serialized = {}
    for key, stats in norm_stats.items():
        serialized[key] = {}
        for name in ("mean", "std", "q01", "q99"):
            value = (
                stats.get(name)
                if isinstance(stats, dict)
                else getattr(stats, name, None)
            )
            if value is not None:
                array = np.asarray(value)
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite OpenPI normalization {key}.{name}")
                serialized[key][name] = array.tolist()
    return {
        "manifest": {
            "asset_ids": asset_ids,
            "config_name": config_name,
            "norm_stats": [str(Path(value) / "norm_stats.json") for value in asset_ids],
            "action_horizon": int(action_horizon),
            "execute_horizon": int(execute_horizon),
            "action_dim": int(action_dim),
            "compatibility": {
                "parameter_precision": parameter_precision,
                "lerobot_compatible": bool(lerobot_compatible),
            },
        },
        "norm_stats": {"norm_stats": serialized},
    }


def _atomic_json_dump(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_checkpoint_assets(assets: dict, save_path: str | Path) -> None:
    """Publish statistics before the manifest in root and actor directories."""
    root = Path(save_path)
    roots = [root, root.parent] if root.name == "actor" else [root, root / "actor"]
    manifest = copy.deepcopy(assets["manifest"])
    for checkpoint_root in roots:
        for name in manifest["norm_stats"]:
            _atomic_json_dump(
                assets["norm_stats"], checkpoint_root / _relative_asset(name)
            )
        _atomic_json_dump(manifest, checkpoint_root / "openpi_assets.json")


def save_checkpoint_assets_distributed(
    assets: dict[str, Any], save_path: str | Path
) -> None:
    """Export on rank zero and propagate its failure before any rank continues."""
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    failure = None
    if rank == 0:
        try:
            export_checkpoint_assets(assets, save_path)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    status = [failure]
    if distributed:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f"Failed exporting OpenPI checkpoint assets: {status[0]}")
