import gc
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from examples.offline_rl.advantage_labeling.recap.process.compute_advantages import (
    _parse_value_model_kwargs,
    compute_advantages_for_dataset,
    load_lerobot_dataset,
)
from rlinf.models.embodiment.value_model.recap.modeling_critic import ValueCriticModel

root = Path('/root/autodl-tmp/Berlin/RLinf')
os.environ['REPO_PATH'] = str(root)
os.environ['RECAP_VALUE_CHECKPOINT'] = '/root/autodl-tmp/outputs/lbench_recap_round1/value/checkpoints/global_step_8000'
with initialize_config_dir(str(root / 'examples/offline_rl/config'), version_base='1.1'):
    cfg = compose(config_name='lbench_recap_advantages')
with open_dict(cfg.advantage):
    cfg.advantage.max_samples = 1024
cfg.advantage.num_dataloader_workers_per_gpu = 8
entry = cfg.data.train_data_paths[0]
model = ValueCriticModel.from_checkpoint(**_parse_value_model_kwargs(cfg), device='cuda:0')
records = []
reference = None
for cache_size in (0, 3):
    dataset, tasks, meta, returns = load_lerobot_dataset(
        Path(entry.dataset_path), returns_tag=cfg.advantage.returns_tag,
        video_decoder_cache_size=cache_size,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    frame = compute_advantages_for_dataset(
        model, dataset, tasks, cfg, OmegaConf.to_container(entry), meta,
        rank=0, world_size=1, global_return_min=-2399, global_return_max=0,
        returns_sidecar=returns,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    record = {'video_decoder_cache_size': cache_size, 'samples': len(frame),
              'seconds': elapsed, 'samples_per_second': len(frame)/elapsed,
              'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
    records.append(record)
    print('BENCHMARK', json.dumps(record), flush=True)
    if reference is None:
        reference = frame
    else:
        pd.testing.assert_frame_equal(reference, frame, check_exact=True)
    del dataset, tasks, meta, returns, frame
    gc.collect()
report = {'runs': records, 'outputs_exactly_equal': True,
          'speedup': records[0]['seconds']/records[1]['seconds'],
          'note': 'Single-GPU sample benchmark with the existing labeling job still running; includes DataLoader startup and temporary result flushing. No production labels written.'}
Path('/root/autodl-tmp/outputs/lbench_recap_preparation/advantage_video_cache_benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
print('RESULT', json.dumps(report), flush=True)
