#!/usr/bin/env bash
# Collect the copy-axis draws behind AI/data/copy_placement_draws/copy_axes_dev5.json.
#
# One process per run, because the axis under test is what a *fresh* process
# does. Four allocator peak levels x four processes = 16 runs, ~4 s each.
#
# Deliberately does NOT set FLYDSL_AUTOTUNE=1: that writes unconditionally to
# ~/.flydsl/autotune/rmsnorm_direct.json and would pollute other agents sharing
# this box.
set -euo pipefail

DEV="${1:-5}"
OUTDIR="${2:-/tmp/copyaxes}"
REPS="${3:-4}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.json

for peak in 0 6 14 20; do
  for i in $(seq 1 "$REPS"); do
    HIP_VISIBLE_DEVICES="$DEV" python "$REPO/AI/probe_copy_size_draws.py" \
      "$OUTDIR/peak${peak}_$i.json" --peak-live-512mib "$peak"
  done
  echo "peak=$peak done ($REPS processes)"
done

python "$REPO/AI/assemble_copy_axes.py" "$OUTDIR"
