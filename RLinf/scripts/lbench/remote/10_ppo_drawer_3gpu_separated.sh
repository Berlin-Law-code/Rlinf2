#!/usr/bin/env bash
set -euo pipefail

# GPU0 trains; GPU1 and GPU2 each run rollout inference and local Isaac workers.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/08_ppo_drawer_2gpu.sh" "$@" --layout separated_3gpu
