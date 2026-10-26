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

"""Real subprocess exit statuses for the embodied entrypoint's exception boundary.

AST extraction executes the production boundary and Cluster atexit/hook methods,
while excluding imports and initialization of Torch, Ray, workers and models.
The Hydra case uses Hydra itself to turn an ordinary task exception into exit 1.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE = textwrap.dedent(
    """
    import ast
    import atexit
    import json
    import os
    from pathlib import Path
    import sys

    repository, mode, apply_boundary = sys.argv[1:]
    repository = Path(repository)
    # Hydra otherwise interprets the probe's own arguments as task overrides.
    sys.argv = [
        'embodied_exit_probe',
        'hydra.run.dir=.',
        'hydra.output_subdir=null',
        'hydra/job_logging=disabled',
        'hydra/hydra_logging=disabled',
    ]
    cluster_path = repository / 'rlinf/scheduler/cluster/cluster.py'
    cluster_ast = ast.parse(cluster_path.read_text())
    cluster_node = next(
        node for node in cluster_ast.body
        if isinstance(node, ast.ClassDef) and node.name == 'Cluster'
    )
    method_names = {'_install_failure_hook', '_shutdown_ray_at_exit'}
    methods = [node for node in cluster_node.body
               if isinstance(node, ast.FunctionDef) and node.name in method_names]
    assert len(methods) == 2
    # Retain the actual staticmethod decorators and bodies. No Ray constructor.
    minimal_class = ast.ClassDef(
        name='Cluster', bases=[], keywords=[], decorator_list=[],
        body=ast.parse('_run_failed = False').body + methods,
    )
    namespace = {'sys': sys, 'os': os, '__name__': '__main__'}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[minimal_class], type_ignores=[]
    )), str(cluster_path), 'exec'), namespace)
    Cluster = namespace['Cluster']
    Cluster._install_failure_hook()
    atexit.register(Cluster._shutdown_ray_at_exit)

    def record_exit_state():
        print('EXIT_PROBE ' + json.dumps({
            'failed': Cluster._run_failed,
            'gpu_frameworks_imported': [
                name for name in ('torch', 'ray', 'tensorflow') if name in sys.modules
            ],
        }), file=sys.stderr, flush=True)

    # Register last so this diagnostic runs before the success fast os._exit.
    atexit.register(record_exit_state)

    def task():
        if mode == 'return':
            return
        if mode == 'exception':
            raise RuntimeError('entrypoint-exception-sentinel')
        codes = {'exit_none': None, 'exit_zero': 0, 'exit_false': False,
                 'exit_seven': 7, 'exit_text': 'explicit-exit-sentinel',
                 'exit_float_zero': 0.0}
        raise SystemExit(codes[mode])

    if mode == 'hydra_exception':
        import hydra

        @hydra.main(version_base='1.1', config_path=None, config_name=None)
        def main(cfg):
            raise RuntimeError('hydra-conversion-sentinel')
    else:
        main = task

    namespace['main'] = main
    if apply_boundary == 'yes':
        entry_path = repository / 'examples/embodiment/train_embodied_agent.py'
        entry_ast = ast.parse(entry_path.read_text())
        entry_block = next(node for node in entry_ast.body
                           if isinstance(node, ast.If)
                           and ast.unparse(node.test) == "__name__ == '__main__'")
        exec(compile(ast.Module(body=[entry_block], type_ignores=[]),
                     str(entry_path), 'exec'), namespace)
    else:
        # Reproduce the old main()-only boundary with the same atexit callback.
        main()
    """
)


def _run_probe(tmp_path, mode, *, apply_boundary=True):
    environment = dict(os.environ)
    environment.pop("HYDRA_FULL_ERROR", None)
    environment.update(
        CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE,
            str(REPO_ROOT),
            mode,
            "yes" if apply_boundary else "no",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    diagnostics = [
        json.loads(line.removeprefix("EXIT_PROBE "))
        for line in result.stderr.splitlines()
        if line.startswith("EXIT_PROBE ")
    ]
    assert len(diagnostics) == 1, result.stdout + result.stderr
    assert diagnostics[0]["gpu_frameworks_imported"] == []
    return result, diagnostics[0]


@pytest.mark.parametrize(
    "mode,expected_code,failed",
    [
        ("return", 0, False),
        ("exit_none", 0, False),
        ("exit_zero", 0, False),
        ("exit_false", 0, False),
        ("exit_seven", 7, True),
        ("exit_text", 1, True),
        # CPython treats a non-integer SystemExit code as an error, even 0.0.
        ("exit_float_zero", 1, True),
        ("exception", 1, True),
        ("hydra_exception", 1, True),
    ],
)
def test_embodied_entrypoint_preserves_real_process_exit_status(
    tmp_path, mode, expected_code, failed
):
    result, state = _run_probe(tmp_path, mode)
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert state["failed"] is failed
    if mode == "exception":
        assert "entrypoint-exception-sentinel" in result.stderr
    if mode == "hydra_exception":
        assert "hydra-conversion-sentinel" in result.stderr


@pytest.mark.parametrize("mode", ["exit_seven", "hydra_exception"])
def test_old_boundary_reproduces_system_exit_masking_by_cluster_fast_exit(
    tmp_path, mode
):
    result, state = _run_probe(tmp_path, mode, apply_boundary=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["failed"] is False
