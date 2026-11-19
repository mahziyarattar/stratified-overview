#!/usr/bin/env bash
# Run a BenchmarkQED judge step with verified limits on the shared endpoint.
#
#   benchmarks/run_judge.sh <config dir> <autoe subcommand> [output dir]
#   benchmarks/run_judge.sh benchmarks/autoe/aa_check pairwise-scores output
#   benchmarks/run_judge.sh benchmarks/autoe/assertions assertion-scores
#
# Loads .env and puts benchmarks/ on PYTHONPATH for the capped client
# (bounded_judge.py: JUDGE_MAX_IN_FLIGHT=14 open requests, JUDGE_MAX_RPM=25).
# Samples the judge's open HTTPS connections every 0.5 s and stops the run if
# they ever exceed JUDGE_MAX_CONNECTIONS (default 16); connections include
# idle ones kept for reuse, hence the gap between the two limits.
# Set JUDGE_EXPECTED_CALLS for a percentage and ETA in run.log and in
# <config dir>/judge_progress.json (updated every 10 calls).
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
config_dir="$1"; subcommand="$2"; output="${3:-}"
cap="${JUDGE_MAX_CONNECTIONS:-16}"
export JUDGE_MAX_IN_FLIGHT="${JUDGE_MAX_IN_FLIGHT:-14}"
export JUDGE_MAX_RPM="${JUDGE_MAX_RPM:-25}" # endpoint key limit: 30/min
log="$config_dir/run.log"

set -a; . "$repo/.env"; set +a
export PYTHONPATH="$repo/benchmarks"

cd "$config_dir"
rm -f judge_progress.json
"$repo/.venv-bench/bin/benchmark-qed" autoe "$subcommand" settings.yaml ${output:+"$output"} > run.log 2>&1 &
job=$!

peak=0
while kill -0 "$job" 2>/dev/null; do
  for pid in $(pgrep -P "$job" 2>/dev/null) "$job"; do
    open=$(lsof -nP -a -p "$pid" -i TCP:443 2>/dev/null | grep -c ESTABLISHED || true)
    if [ "$open" -gt "$peak" ]; then peak=$open; fi
    if [ "$open" -gt "$cap" ]; then
      echo "ABORT: $open open connections exceed the cap of $cap" | tee -a run.log
      kill "$job"; wait "$job" 2>/dev/null || true
      exit 2
    fi
  done
  sleep 0.5
done

status=0; wait "$job" || status=$?
echo "exit $status | peak open connections: $peak (hard stop above $cap)" | tee -a run.log
exit "$status"
