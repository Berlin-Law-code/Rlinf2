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

"""Preserve a LeRobot Pi0.5 FP32 base as a validated OpenPI FP32 asset.

Only CPU tensors are used. A reference OpenPI safetensors header defines the
required keys and shapes; its tensor payload is never loaded. Existing output
directories are rejected, and BF16 inputs are rejected instead of upcasting
already-rounded weights and presenting them as the original FP32 base.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
from collections.abc import Mapping
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_PALI_LM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
_PALI_EMBEDDING = (
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
)


def _weight_path(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().resolve()
    return path / "model.safetensors" if path.is_dir() else path


def read_header(path: pathlib.Path) -> dict[str, Any]:
    """Read only the safetensors header, without materializing any weights."""
    with path.open("rb") as source:
        size_bytes = source.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > 100_000_000:
            raise ValueError(f"Safetensors header is unexpectedly large: {path}")
        return json.loads(source.read(size))


def _canonical_key(key: str) -> str:
    key = key.removeprefix("model.")
    return _PALI_LM_HEAD if key == _PALI_EMBEDDING else key


def build_key_mapping(
    source: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Validate exact source coverage, allowing only the known tied embedding."""
    source_keys: dict[str, list[str]] = {}
    for key, spec in source.items():
        if key == "__metadata__":
            continue
        if spec["dtype"] != "F32":
            raise ValueError(
                f"Source weights must be original FP32, got {spec['dtype']} for {key}"
            )
        source_keys.setdefault(_canonical_key(key), []).append(key)

    reference_keys = {
        _canonical_key(key): key for key in reference if key != "__metadata__"
    }
    missing = sorted(set(reference_keys) - set(source_keys))
    unexpected = sorted(set(source_keys) - set(reference_keys))
    if missing or unexpected:
        raise ValueError(
            "Source/reference keys do not match: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    mapping: dict[str, list[str]] = {}
    for canonical, target in reference_keys.items():
        keys = source_keys[canonical]
        for key in keys:
            if source[key]["shape"] != reference[target]["shape"]:
                raise ValueError(
                    f"Shape mismatch for {key}: {source[key]['shape']} "
                    f"!= {reference[target]['shape']}"
                )
        mapping[target] = keys
    return mapping


def convert_base(
    source: pathlib.Path, reference: pathlib.Path, output_dir: pathlib.Path
) -> dict[str, Any]:
    """Write a new FP32 asset after validating source keys, shapes and dtypes."""
    source = _weight_path(source)
    reference = _weight_path(reference)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    source_header = read_header(source)
    reference_header = read_header(reference)
    mapping = build_key_mapping(source_header, reference_header)

    config_path = reference.parent / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["precision"] = "float32"
    config["parameter_precision"] = "float32"

    weights = {}
    with safe_open(source, framework="pt", device="cpu") as source_file:
        for target, source_keys in mapping.items():
            tensor = source_file.get_tensor(source_keys[0])
            for alias in source_keys[1:]:
                if not torch.equal(tensor, source_file.get_tensor(alias)):
                    raise ValueError(f"Tied source tensors differ: {source_keys}")
            weights[target] = tensor

        metadata = dict(reference_header.get("__metadata__", {}))
        metadata["parameter_precision"] = "float32"
        # mkdir(exist_ok=False) also protects against concurrent writers.
        output_dir.mkdir(parents=True, exist_ok=False)
        save_file(weights, str(output_dir / "model.safetensors"), metadata=metadata)

    provenance = {
        "source": str(source),
        "reference": str(reference),
        "source_dtype": "float32",
        "output_dtype": "float32",
        "tensor_count": len(weights),
        "source_size_bytes": source.stat().st_size,
        "reference_size_bytes": reference.stat().st_size,
        "renamed_keys": {
            target: keys for target, keys in mapping.items() if keys != [target]
        },
    }
    for name, data in (("config.json", config), ("conversion.json", provenance)):
        with (output_dir / name).open("x", encoding="utf-8") as target_file:
            json.dump(data, target_file, indent=2, ensure_ascii=False)
            target_file.write("\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(convert_base(args.source, args.reference, args.output_dir), indent=2)
    )


if __name__ == "__main__":
    main()
