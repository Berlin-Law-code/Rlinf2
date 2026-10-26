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

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "convert_pi05_base_to_openpi",
    ROOT / "toolkits/lerobot/convert_pi05_base_to_openpi.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_CONVERTER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONVERTER)


def _write_weights(path, weights, metadata=None):
    path.mkdir()
    save_file(weights, str(path / "model.safetensors"), metadata=metadata)
    (path / "config.json").write_text('{"pi05": true, "precision": "bfloat16"}')
    return path


def test_base_conversion_preserves_low_bits_and_maps_embedding_alias(tmp_path):
    values = torch.tensor([[1.0001, 1.0002]])
    source = _write_weights(
        tmp_path / "source",
        {"model." + _CONVERTER._PALI_EMBEDDING: values},
    )
    reference = _write_weights(
        tmp_path / "reference",
        {_CONVERTER._PALI_LM_HEAD: values.to(torch.bfloat16)},
        {_CONVERTER._PALI_EMBEDDING: _CONVERTER._PALI_LM_HEAD},
    )
    output = tmp_path / "output"
    provenance = _CONVERTER.convert_base(source, reference, output)
    result = load_file(str(output / "model.safetensors"))
    assert torch.equal(result[_CONVERTER._PALI_LM_HEAD], values)
    assert result[_CONVERTER._PALI_LM_HEAD].dtype == torch.float32
    assert provenance["tensor_count"] == 1
    assert json.loads((output / "config.json").read_text())["precision"] == "float32"
    assert (
        _CONVERTER.read_header(output / "model.safetensors")["__metadata__"][
            _CONVERTER._PALI_EMBEDDING
        ]
        == _CONVERTER._PALI_LM_HEAD
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _CONVERTER.convert_base(source, reference, output)
    assert torch.equal(
        load_file(str(output / "model.safetensors"))[_CONVERTER._PALI_LM_HEAD], values
    )


@pytest.mark.parametrize(
    ("source_spec", "error"),
    [
        ({"weight": {"dtype": "BF16", "shape": [2]}}, "original FP32"),
        ({"weight": {"dtype": "F32", "shape": [3]}}, "Shape mismatch"),
        ({"other": {"dtype": "F32", "shape": [2]}}, "keys do not match"),
    ],
)
def test_base_conversion_rejects_rounded_or_incompatible_sources(source_spec, error):
    with pytest.raises(ValueError, match=error):
        _CONVERTER.build_key_mapping(
            source_spec, {"weight": {"dtype": "BF16", "shape": [2]}}
        )


def test_conflicting_tied_values_fail_before_creating_output(tmp_path):
    source = _write_weights(
        tmp_path / "source",
        {
            _CONVERTER._PALI_EMBEDDING: torch.ones(1, 2),
            _CONVERTER._PALI_LM_HEAD: torch.zeros(1, 2),
        },
    )
    reference = _write_weights(
        tmp_path / "reference", {_CONVERTER._PALI_LM_HEAD: torch.ones(1, 2)}
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Tied source tensors differ"):
        _CONVERTER.convert_base(source, reference, output)
    assert not output.exists()
