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

import copy
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from openpi import transforms
from openpi.models.model import ModelType, Observation
from openpi.shared.normalize import NormStats
from openpi.training import data_loader
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings

from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
    build_official_openpi_sft_dataloader,
)
from rlinf.models.embodiment import openpi as model_factory
from rlinf.models.embodiment.openpi import dataconfig, openpi_action_model, precision
from rlinf.models.embodiment.openpi.dataconfig import lbench_dataconfig
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)


@dataclasses.dataclass
class _TrainConfig:
    data: object
    model: object
    assets_dirs: object
    num_workers: int = 0
    seed: int = 0


class _WrapperOnlyModel:
    """Run the real get_model wrapper assembly without allocating model weights."""

    def __init__(self, config):
        self.config = config

    def setup_wrappers(self, transforms, output_transforms):
        self.inputs = transforms
        self.outputs = output_transforms


class _LocalTokenizePrompt:
    """Keep the model input schema without downloading a tokenizer in tests."""

    def __call__(self, data):
        data.pop("prompt")
        data["tokenized_prompt"] = np.array([1, 2], dtype=np.int64)
        data["tokenized_prompt_mask"] = np.ones(2, dtype=bool)
        return data


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    import openpi.shared.download as download
    import safetensors.torch

    stats = NormStats(
        mean=np.zeros(8, dtype=np.float32),
        std=np.ones(8, dtype=np.float32),
        q01=np.array([-1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, 0], np.float32),
        q99=np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1], np.float32),
    )
    norm_stats = {"state": stats, "actions": stats}
    # The checkpoint loader now validates explicit assets before allocating a
    # model. Materialize the same statistics used by the transform fixture.
    (tmp_path / "norm_stats.json").write_text(
        json.dumps(
            {
                "norm_stats": {
                    key: {
                        field: getattr(value, field).tolist()
                        for field in ("mean", "std", "q01", "q99")
                    }
                    for key, value in norm_stats.items()
                }
            }
        )
    )
    monkeypatch.setattr(
        lbench_dataconfig.LeRobotLBenchJointDataConfig,
        "_load_norm_stats",
        lambda *args: norm_stats,
    )
    # Tokenizer downloads and network weights are irrelevant to the image/norm
    # contract; retain the actual resize and padding factory stages.
    monkeypatch.setattr(
        lbench_dataconfig,
        "ModelTransformFactory",
        lambda: (
            lambda model: transforms.Group(
                inputs=[
                    transforms.ResizeImages(224, 224),
                    _LocalTokenizePrompt(),
                    transforms.PadStatesAndActions(32),
                ]
            )
        ),
    )
    monkeypatch.setattr(download, "maybe_download", lambda path: path)
    monkeypatch.setattr(openpi_action_model, "OpenPi0Config", SimpleNamespace)
    monkeypatch.setattr(
        openpi_action_model, "OpenPi0ForRLActionPrediction", _WrapperOnlyModel
    )
    monkeypatch.setattr(safetensors.torch, "load_model", lambda *a, **kw: ([], []))
    monkeypatch.setattr(precision, "apply_parameter_precision", lambda *args: None)

    rng = np.random.default_rng(31)
    sample = {
        "observation.state": np.array([-0.1] * 7 + [1.0], dtype=np.float32),
        "action": np.tile(np.array([0.11] * 7 + [0], np.float32), (50, 1)),
        "prompt": "put the object into the drawer",
        **{
            f"observation.images.{camera}": rng.random((7, 11, 3), dtype=np.float32)
            for camera in ("obs_camera", "obs_camera_2", "realsense")
        },
    }

    def build(compatible):
        factory = lbench_dataconfig.LeRobotLBenchJointDataConfig(
            repo_id="test/lbench", lerobot_compatible=compatible
        )
        model = SimpleNamespace(
            model_type=ModelType.PI05,
            action_dim=32,
            action_horizon=50,
            action_chunk=20,
            action_env_dim=8,
            train_expert_only=False,
            add_value_head=False,
        )
        config = _TrainConfig(factory, model, tmp_path)
        monkeypatch.setattr(dataconfig, "get_openpi_config", lambda *a, **kw: config)
        captured = {}

        def create_loader(config, **kwargs):
            captured.update(kwargs)
            created = config.data.create(config.assets_dirs, config.model)
            dataset = data_loader.transform_dataset(
                [copy.deepcopy(sample)],
                created,
                skip_norm_stats=kwargs["skip_norm_stats"],
            )
            return SimpleNamespace(
                dataset=dataset,
                data_config=lambda: created,
                _data_loader=SimpleNamespace(
                    _data_loader=SimpleNamespace(batch_size=4)
                ),
            )

        monkeypatch.setattr(data_loader, "create_data_loader", create_loader)
        cfg = OmegaConf.create(
            {
                "actor": {
                    "micro_batch_size": 4,
                    "seed": 1000,
                    "model": {
                        "model_type": "openpi",
                        "model_path": str(tmp_path),
                        "openpi": {"config_name": "pi05_lbench_joint"},
                        "openpi_data": {
                            "lerobot_compatible": compatible,
                            "norm_stats_path": str(tmp_path / "norm_stats.json"),
                        },
                    },
                },
                "data": {"num_workers": 0},
            }
        )
        loader, created = build_official_openpi_sft_dataloader(
            cfg, world_size=1, rank=0, data_paths="test/lbench"
        )
        wrapped = model_factory.get_model(cfg.actor.model)
        return loader, created, wrapped, captured

    return SimpleNamespace(build=build, sample=sample, stats=norm_stats)


@pytest.mark.parametrize("compatible", [False, True])
def test_sft_and_eval_preprocessing_match_without_double_normalization(
    pipeline, compatible
):
    loader, config, model, captured = pipeline.build(compatible)
    assert captured["skip_norm_stats"] is compatible
    trained = loader.dataset[0]
    eval_input = transforms.compose(config.repack_transforms.inputs)(
        copy.deepcopy(pipeline.sample)
    )
    evaluated = transforms.compose(model.inputs)(eval_input)
    for key in ("state", "actions"):
        np.testing.assert_array_equal(trained[key], evaluated[key])
        raw = pipeline.sample["observation.state" if key == "state" else "action"]
        if compatible:
            stats = pipeline.stats[key]
            low, high = torch.from_numpy(stats.q01), torch.from_numpy(stats.q99)
            expected = (2 * (torch.from_numpy(raw) - low) / (high - low) - 1).numpy()
        else:
            expected = transforms.Normalize(pipeline.stats, use_quantiles=True)(
                {key: raw}
            )[key]
        np.testing.assert_array_equal(trained[key][..., :8], expected)
        np.testing.assert_array_equal(trained[key][..., 8:], 0)
    for key in trained["image"]:
        np.testing.assert_array_equal(trained["image"][key], evaluated["image"][key])
        expected_shape = (3, 224, 224) if compatible else (224, 224, 3)
        assert trained["image"][key].shape == expected_shape
        assert trained["image"][key].dtype == (np.float32 if compatible else np.uint8)


@pytest.mark.parametrize("compatible", [False, True])
def test_sft_and_eval_images_reach_siglip_as_bchw(pipeline, compatible):
    """Exercise the dtype-dependent Observation boundary, not just transforms."""
    loader, config, wrapped, _ = pipeline.build(compatible)
    batch_size = 2
    torch_loader = data_loader.TorchDataLoader(
        [loader.dataset[0] for _ in range(batch_size)],
        local_batch_size=batch_size,
        num_batches=1,
        num_workers=0,
        framework="pytorch",
    )
    train_observation, _ = next(iter(data_loader.DataLoaderImpl(config, torch_loader)))

    # Use the real evaluation wrapper and preprocessing methods, without
    # allocating PaliGemma or the action expert.
    runtime = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    torch.nn.Module.__init__(runtime)
    runtime.config = SimpleNamespace(sft_image_augmentation=False)
    runtime.setup_wrappers(wrapped.inputs, wrapped.outputs)
    repacked = transforms.compose(config.repack_transforms.inputs)(
        copy.deepcopy(pipeline.sample)
    )
    eval_batch = {
        key: [value] * batch_size
        if isinstance(value, str)
        else np.stack([value] * batch_size)
        for key, value in repacked.items()
    }
    eval_observation = Observation.from_dict(
        runtime.input_transform(eval_batch, transpose=False)
    )
    with runtime._sft_observation_preprocessing():
        train_images, *_ = runtime._preprocess_observation(
            train_observation, train=True
        )
    eval_images, *_ = runtime._preprocess_observation(eval_observation, train=False)

    vision = SiglipVisionEmbeddings(
        SiglipVisionConfig(image_size=224, patch_size=14, hidden_size=16)
    )
    losses = []
    for key, trained, evaluated in zip(
        train_observation.images, train_images, eval_images, strict=True
    ):
        # Reach the same real Conv2d that crashed, including its backward path.
        embedded = vision(trained)
        assert embedded.shape == (batch_size, 256, 16)
        assert trained.shape == (batch_size, 3, 224, 224)
        assert trained.dtype == torch.float32
        torch.testing.assert_close(trained, evaluated, rtol=0, atol=0)
        torch.testing.assert_close(
            trained, train_observation.images[key], rtol=0, atol=0
        )
        torch.testing.assert_close(embedded, vision(evaluated), rtol=0, atol=0)
        losses.append(embedded.square().mean())
    sum(losses).backward()
    assert vision.patch_embedding.weight.grad is not None
    assert torch.isfinite(vision.patch_embedding.weight.grad).all()


@pytest.mark.parametrize("compatible", [False, True])
def test_eval_unnormalizes_actions_exactly_once_and_slices_padding(
    pipeline, compatible
):
    _, _, model, _ = pipeline.build(compatible)
    normalized = np.linspace(-2, 2, 50 * 32, dtype=np.float32).reshape(50, 32)
    outputs = {"actions": normalized.copy(), "state": np.zeros(32, np.float32)}
    actual = transforms.compose(model.outputs)(copy.deepcopy(outputs))["actions"]
    if compatible:
        stats = pipeline.stats["actions"]
        low, high = torch.from_numpy(stats.q01), torch.from_numpy(stats.q99)
        expected = (torch.from_numpy(normalized[:, :8]) + 1) * (high - low) / 2 + low
        expected = expected.numpy()
    else:
        expected = transforms.Unnormalize(pipeline.stats, use_quantiles=True)(outputs)[
            "actions"
        ][:, :8]
    assert actual.shape == (50, 8)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("location", ["openpi_assets.json", "actor/openpi_assets.json"])
def test_checkpoint_compatibility_manifest_is_found_at_both_save_layouts(
    tmp_path, location
):
    settings = {"lerobot_compatible": True, "parameter_precision": "float32"}
    manifest = tmp_path / location
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"compatibility": settings}), encoding="utf-8")
    assert model_factory._checkpoint_compatibility_settings(str(tmp_path)) == settings


def test_legacy_checkpoint_without_manifest_keeps_default_contract(tmp_path):
    assert model_factory._checkpoint_compatibility_settings(str(tmp_path)) == {}


def test_sft_asset_export_roundtrips_stats_and_compatibility(monkeypatch, tmp_path):
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    source = tmp_path / "source_stats.json"
    source.write_text('{"norm_stats": {"actions": {"q01": [0.0]}}}', encoding="utf-8")
    monkeypatch.setattr(
        dataconfig,
        "get_openpi_config",
        lambda *args: SimpleNamespace(data=SimpleNamespace(repo_id="canonical_asset")),
    )
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "actor": {
                    "model": {
                        "num_action_chunks": 50,
                        "action_dim": 8,
                        "openpi": {
                            "config_name": "pi05_lbench_joint",
                            "parameter_precision": "float32",
                        },
                        "openpi_data": {"lerobot_compatible": True},
                    }
                }
            }
        ),
        data_config=SimpleNamespace(asset_id="runtime_asset"),
        _atomic_copy_file=FSDPVlaSftWorker._atomic_copy_file,
        _atomic_json_dump=FSDPVlaSftWorker._atomic_json_dump,
    )
    checkpoint = tmp_path / "global_step_5000"
    FSDPVlaSftWorker._export_openpi_assets(
        worker, str(checkpoint / "actor"), str(source)
    )
    for root in (checkpoint, checkpoint / "actor"):
        assert model_factory._checkpoint_compatibility_settings(str(root)) == {
            "parameter_precision": "float32",
            "lerobot_compatible": True,
        }
        for asset_id in ("canonical_asset", "runtime_asset"):
            assert (
                root / asset_id / "norm_stats.json"
            ).read_bytes() == source.read_bytes()
        manifest = json.loads((root / "openpi_assets.json").read_text())
        assert manifest["action_horizon"] == 50
        assert manifest["action_dim"] == 8
