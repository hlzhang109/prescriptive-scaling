#!/usr/bin/env bash
# Regenerate paper-used artifacts from scratch (without using regenerate_all.sh).
#
# Primary target:
#   - outputs/figures_main_paper/*
#
# Supporting targets (used by paper figures and checks):
#   - outputs/sigmoid/period4/single_k/plots
#   - outputs/sigmoid/period4/single_k/new_models/plots
#   - outputs/sigmoid/period4/single_k/new_models_p5/plots
#   - outputs/evaluation/sigmoid/period4/single_k/no_budget/plots
#   - evaluation_pinball/period4_singlek_no_budget/plots
#   - outputs/sigmoid_size/period4/single_k/plots
#   - outputs/sweeps_fullrange
#   - outputs_old/sigmoid/period4/single_k/plots

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DO_CLEAN=1
RUN_SWEEPS=1

usage() {
  cat <<EOF
Usage: ./regenerate_paper_artifacts_from_scratch.sh [OPTIONS]

Options:
  --python <exe>   Python executable to use (default: ${PYTHON_BIN})
  --no-clean       Do not remove target output folders before regeneration
  --skip-sweeps    Skip outputs/sweeps_fullrange regeneration (Figure 6 may fail if absent)
  -h, --help       Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-clean)
      DO_CLEAN=0
      shift
      ;;
    --skip-sweeps)
      RUN_SWEEPS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

run_cmd() {
  echo
  echo "[RUN] $*"
  "$@"
}

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] Missing required file: $p" >&2
    exit 1
  fi
}

require_nonempty_dir() {
  local p="$1"
  if [[ ! -d "$p" ]]; then
    echo "[ERROR] Missing directory: $p" >&2
    exit 1
  fi
  if [[ -z "$(find "$p" -type f -print -quit 2>/dev/null)" ]]; then
    echo "[ERROR] Directory exists but has no files: $p" >&2
    exit 1
  fi
}

make_size_proxy_csv() {
  local in_csv="$1"
  local out_csv="$2"
  "$PYTHON_BIN" - "$in_csv" "$out_csv" <<'PY'
import sys
import pandas as pd

in_csv, out_csv = sys.argv[1], sys.argv[2]
df = pd.read_csv(in_csv)
if "#Params (B)" not in df.columns:
    raise SystemExit(f"Missing '#Params (B)' column in {in_csv}")

# Map compute formula to size so the refactored smoother can still fit size-axis frontiers:
# compute = 6 * tokens_proxy * params_proxy = params_B.
df["_size_tokens_proxy"] = 1.0 / 6.0
df["_size_params_proxy"] = pd.to_numeric(df["#Params (B)"], errors="coerce")
df.to_csv(out_csv, index=False)
PY
}

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache}"
mkdir -p "$MPLCONFIGDIR"

cd "$ROOT_DIR"

echo "[INFO] Root: $ROOT_DIR"
echo "[INFO] Python: $PYTHON_BIN"

# ---------------------------------------------------------------------------
# 0) Preflight checks
# ---------------------------------------------------------------------------
require_file "tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv"
require_file "tables/open_llm_leaderboard/open_llm_leaderboard_old_with_tokens.csv"
require_file "tables/open_llm_leaderboard/validation_leaderboard.csv"
require_file "tables/open_llm_leaderboard/new_eval_leaderboard.csv"
require_file "tables/top_models_by_base.csv"
require_file "tables/new_leaderboard_results_with_tokens.csv"
require_file "tables/additional_post_trained_models.csv"
require_file "tables/artificial_analysis/aime-2025.csv"
require_file "tables/artificial_analysis/math-500.csv"
require_file "outputs_epoch_ai_benchmarks_runs/sigmoid/no_split/points/GPQA_diamond.csv"
require_file "outputs_epoch_ai_benchmarks_runs/sigmoid/no_split/curves/GPQA_diamond.csv"

# ---------------------------------------------------------------------------
# 1) Clean targets (optional, default ON)
# ---------------------------------------------------------------------------
if [[ "$DO_CLEAN" -eq 1 ]]; then
  echo "[INFO] Cleaning target output directories..."
  rm -rf \
    outputs/figures_main_paper \
    outputs/sigmoid/period4/single_k \
    outputs/evaluation/sigmoid/period4/single_k/no_budget \
    evaluation_pinball/period4_singlek_no_budget \
    outputs/sigmoid_size/period4/single_k \
    outputs/sweeps_fullrange \
    outputs_old/sigmoid/period4/single_k \
    outputs/sigmoid/no_split/pretrain_vs_posttrain \
    outputs/open_llm_leaderboard/current
fi

# ---------------------------------------------------------------------------
# 2) Main OLL period4/single_k frontiers
# ---------------------------------------------------------------------------
run_cmd "$PYTHON_BIN" scripts/smooth_single_skill_frontier.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --sigmoid \
  --split_mode period4 \
  --period4_train_mode single_k \
  --tau 0.98 \
  --out_dir outputs/sigmoid/period4/single_k

# ---------------------------------------------------------------------------
# 3) Period4 old-vs-new overlays (new_models / new_models_p5)
# ---------------------------------------------------------------------------
run_cmd "$PYTHON_BIN" scripts/plot/overlays/plot_period4_frontiers_old_vs_new.py \
  --oll_csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --new_eval_csv tables/open_llm_leaderboard/validation_leaderboard.csv \
  --new_compute_csv tables/open_llm_leaderboard/new_eval_leaderboard.csv \
  --top_models_csv tables/top_models_by_base.csv \
  --mode same_period \
  --tau 0.98 \
  --lambda_b 1e-3 \
  --scatter_style two_color \
  --out_dir outputs/sigmoid/period4/single_k/new_models

run_cmd "$PYTHON_BIN" scripts/plot/overlays/plot_period4_frontiers_old_vs_new.py \
  --oll_csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --new_eval_csv tables/open_llm_leaderboard/validation_leaderboard.csv \
  --new_compute_csv tables/open_llm_leaderboard/new_eval_leaderboard.csv \
  --top_models_csv tables/top_models_by_base.csv \
  --mode p4_to_p5 \
  --tau 0.98 \
  --lambda_b 1e-3 \
  --scatter_style two_color \
  --extra_points_csv tables/new_leaderboard_results_with_tokens.csv \
  --out_dir outputs/sigmoid/period4/single_k/new_models_p5

# ---------------------------------------------------------------------------
# 4) Evaluation plots (coverage + pinball, period4/single_k/no_budget)
# ---------------------------------------------------------------------------
run_cmd "$PYTHON_BIN" scripts/evaluate/sigmoid_binned_mae.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --split_mode period4 \
  --train_mode single_k \
  --tau 0.98 \
  --bins 10 \
  --min_bin_size 30 \
  --oos_bins train_overlap \
  --out_base outputs/evaluation/sigmoid/period4/single_k/no_budget

run_cmd "$PYTHON_BIN" scripts/plot/diagnostics/eval_sigmoid.py \
  --period4_singlek_base outputs/evaluation/sigmoid/period4/single_k/no_budget \
  --make_calib_summary_panels

run_cmd "$PYTHON_BIN" scripts/evaluate/sigmoid_binned_pinball.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --tau 0.98 \
  --bins 10 \
  --min_bin_size 30 \
  --oos_bins train_overlap \
  --out_base evaluation_pinball/period4_singlek_no_budget

run_cmd "$PYTHON_BIN" scripts/plot/diagnostics/eval_sigmoid.py \
  --period4_singlek_base evaluation_pinball/period4_singlek_no_budget \
  --make_calib_summary_panels

# ---------------------------------------------------------------------------
# 5) Size-axis period4/single_k frontiers + style restyle
# ---------------------------------------------------------------------------
TMP_WORK="$(mktemp -d "${TMPDIR:-/tmp}/skill_frontier_size_proxy.XXXXXX")"
trap 'rm -rf "$TMP_WORK"' EXIT

SIZE_PROXY_MAIN="$TMP_WORK/open_llm_with_size_proxy.csv"
make_size_proxy_csv "tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv" "$SIZE_PROXY_MAIN"

run_cmd "$PYTHON_BIN" scripts/smooth_single_skill_frontier.py \
  --csv "$SIZE_PROXY_MAIN" \
  --compute_product_cols "_size_tokens_proxy" "_size_params_proxy" \
  --compute_multiplier 6.0 \
  --sigmoid \
  --split_mode period4 \
  --period4_train_mode single_k \
  --tau 0.98 \
  --out_dir outputs/sigmoid_size/period4/single_k

run_cmd "$PYTHON_BIN" scripts/plot/restyle/restyle_sigmoid_size_period4_singlek_plots.py

# ---------------------------------------------------------------------------
# 6) Alpha sweeps (for outputs/sweeps_fullrange and Figure 6)
# ---------------------------------------------------------------------------
if [[ "$RUN_SWEEPS" -eq 1 ]]; then
  run_cmd "$PYTHON_BIN" scripts/evaluate/sweep_alpha.py \
    --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
    --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
    --compute_multiplier 6.0 \
    --alphas 5 10 20 50 100 \
    --design_objective i_optimal_predvar_balanced \
    --balance_lambda 1e-3 \
    --exchange_passes 2 \
    --write_pinball \
    --out_dir outputs/sweeps_fullrange
else
  echo "[INFO] Skipping sweeps_fullrange (--skip-sweeps)."
fi

# ---------------------------------------------------------------------------
# 7) Old OLL period4/single_k (then restyle period symbol to t)
# ---------------------------------------------------------------------------
run_cmd "$PYTHON_BIN" scripts/smooth_single_skill_frontier.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_old_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --sigmoid \
  --split_mode period4 \
  --period4_train_mode single_k \
  --tau 0.98 \
  --out_dir outputs_old/sigmoid/period4/single_k

run_cmd "$PYTHON_BIN" scripts/plot/restyle/restyle_outputs_old_sigmoid_period4_singlek_plots.py \
  --period_symbol t

# ---------------------------------------------------------------------------
# 8) Additional inputs needed by outputs/figures_main_paper
# ---------------------------------------------------------------------------
# Figure 4 / Figure 9 dependency
run_cmd "$PYTHON_BIN" scripts/run/pretrain_vs_posttrain.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --compare pretrain_vs_posttrain \
  --x_axis compute \
  --tau 0.98 \
  --lambda_b 1e-3 \
  --out_dir outputs/sigmoid/no_split/pretrain_vs_posttrain

# Figure 7 dependency (monthly dominance metrics)
run_cmd "$PYTHON_BIN" scripts/open_llm_leaderboard/open_llm_leaderboard_v2_scaling.py \
  --no-fetch_missing_last_modified \
  --out_dir outputs/open_llm_leaderboard

# ---------------------------------------------------------------------------
# 9) Paper figures
#    figure1 -> scripts/plot/paper/figure1_main_paper.py
#    figure3 -> scripts/plot/paper/figure3_main_paper.py
#    figure4 -> scripts/plot/paper/figure4/plot_fig4.py
#    figure5 -> scripts/plot/paper/figure5_main_paper.py
#    figure6 -> scripts/plot/paper/figure6_main_paper.py
#    figure7 -> scripts/plot/paper/figure7_main_paper.py
#    figure7a/7b -> scripts/plot/paper/figure7ab_main_paper.py
#    figure9 -> scripts/plot/paper/figure9_main_paper.py
# ---------------------------------------------------------------------------
run_cmd "$PYTHON_BIN" scripts/plot/paper/figure1_main_paper.py
run_cmd "$PYTHON_BIN" scripts/plot/paper/figure3_main_paper.py
run_cmd "$PYTHON_BIN" scripts/plot/paper/figure4/plot_fig4.py
run_cmd "$PYTHON_BIN" scripts/plot/paper/figure5_main_paper.py

if [[ "$RUN_SWEEPS" -eq 1 || -d outputs/sweeps_fullrange ]]; then
  run_cmd "$PYTHON_BIN" scripts/plot/paper/figure6_main_paper.py
else
  echo "[WARN] Skipping Figure 6 (outputs/sweeps_fullrange not available)."
fi

run_cmd env PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/plot/paper/figure7_main_paper.py
run_cmd "$PYTHON_BIN" scripts/plot/paper/figure7ab_main_paper.py
run_cmd "$PYTHON_BIN" scripts/plot/paper/figure9_main_paper.py

# Keep legacy preview filename present.
if [[ -f outputs/figures_main_paper/figure5_main_paper.png ]]; then
  cp -f outputs/figures_main_paper/figure5_main_paper.png \
    outputs/figures_main_paper/figure5_main_paper_preview.png
fi

# ---------------------------------------------------------------------------
# 10) Output checks
# ---------------------------------------------------------------------------
require_nonempty_dir "outputs/figures_main_paper"
require_nonempty_dir "outputs/sigmoid/period4/single_k/plots"
require_nonempty_dir "outputs/sigmoid/period4/single_k/new_models/plots"
require_nonempty_dir "outputs/sigmoid/period4/single_k/new_models_p5/plots"
require_nonempty_dir "outputs/evaluation/sigmoid/period4/single_k/no_budget/plots"
require_nonempty_dir "evaluation_pinball/period4_singlek_no_budget/plots"
require_nonempty_dir "outputs/sigmoid_size/period4/single_k/plots"
require_nonempty_dir "outputs_old/sigmoid/period4/single_k/plots"
require_nonempty_dir "outputs/sigmoid/no_split/pretrain_vs_posttrain/plots"
require_nonempty_dir "outputs/open_llm_leaderboard/current"
if [[ "$RUN_SWEEPS" -eq 1 ]]; then
  require_nonempty_dir "outputs/sweeps_fullrange"
fi

echo
echo "[DONE] Paper artifacts regenerated successfully."
