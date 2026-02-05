#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

Rscript scripts/update_season.R "$@"

# Run analysis scripts to refresh figures/reports after data update.
# Set SKIP_ANALYSIS=1 to disable.
if [[ "${SKIP_ANALYSIS:-0}" != "1" ]]; then
  if [[ -x "$repo_root/.venv/bin/python" ]]; then
    PYTHON_BIN="$repo_root/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Python not found; skipping analysis scripts."
    exit 0
  fi

  export SPATIALSPORTSR_DB_PATH="$repo_root/data/parsed/nba.sqlite"
  export MPLBACKEND="${MPLBACKEND:-Agg}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/spatialsportsr-mpl}"
  mkdir -p "$MPLCONFIGDIR"

  run_py() {
    local script="$1"
    echo "Running ${script}..."
    if ! "$PYTHON_BIN" "$repo_root/$script"; then
      echo "Warning: ${script} failed; continuing."
    fi
  }

  run_py "analysis/expected_points_analysis.py"
  run_py "analysis/advanced_analytics.py"
  run_py "analysis/player_performance_analysis.py"

  if [[ -f "$repo_root/analysis/data/player_salaries_2024-25.csv" ]]; then
    run_py "analysis/value_analysis.py"
  else
    echo "Skipping value_analysis.py (missing analysis/data/player_salaries_2024-25.csv)."
  fi

  run_py "analysis/shot_density.py"

  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1; then
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pygam") else 1)
PY
    run_py "analysis/gam_analysis.py"
  else
    echo "Skipping gam_analysis.py (pygam not installed)."
  fi
fi
