#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
run_dir="$root_dir/guinea-pig/runs/10tev_ffs_xsuite_no_sr"
guinea_bin="$root_dir/guinea-pig/install/bin/guinea"

if [[ ! -x "$guinea_bin" ]]; then
  printf 'GUINEA-PIG executable not found: %s\n' "$guinea_bin" >&2
  exit 1
fi

"$root_dir/../.venv/bin/python" "$root_dir/prepare_10tev_ffs_guinea_pig.py" \
  --source "$root_dir/ffs_10tev_tracking_no_sr_start_end.npz" \
  --run-dir "$run_dir"

cd "$run_dir"
"$guinea_bin" \
  --acc_file=guinea_10tev_ffs.acc \
  --el_file=electron.ini \
  --pos_file=positron.ini \
  ffs10tev xsuite_endpoint luminosity.out

test -s luminosity.out
grep '^lumi_ee=' luminosity.out
