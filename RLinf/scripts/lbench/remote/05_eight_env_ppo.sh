#!/usr/bin/env bash
set -euo pipefail
exec "${RLINF_PYTHON:-python}" "$(dirname "$0")/../../../toolkits/lbench_remote_stage.py" ppo-8 "$@"
