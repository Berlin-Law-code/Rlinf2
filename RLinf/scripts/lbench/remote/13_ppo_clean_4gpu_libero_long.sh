#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clean_task="clean_the_dirt_with_browncloth_lighttable_part0"
for argument in "$@"; do
  if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
    printf 'Default task: %s; four GPUs, 600 real episode steps.\n\n' "$clean_task"
    break
  fi
done
# Put the default first so an explicit --task later in the command still wins.
exec bash "$script_dir/11_ppo_drawer_4gpu_libero_long.sh" --task "$clean_task" "$@"
