# LLM Skill Frontiers (Compute → Performance)

This repo estimates and evaluates **performance frontiers** for LLM benchmarks as a function of pretraining compute (and optionally model size). It also supports **budgeted evaluation designs** (D‑optimal and I‑optimal variants), plus alpha sweeps and plotting utilities.

If you only want to regenerate results, start with `./regenerate_all.sh` (or individual sections).  
If you specifically want the paper-used artifacts only, use `./regenerate_paper_artifacts_from_scratch.sh`.

## Inputs (data sources)

### Open LLM Leaderboard (primary)
- File: `tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv`
- Compute proxy (default): `6 * ("Pretraining tokens (T)") * ("#Params (B)")`
- Main 6 benchmarks use **raw** accuracies:
  - `IFEval Raw`, `BBH Raw`, `MATH Lvl 5 Raw`, `GPQA Raw`, `MUSR Raw`, `MMLU-PRO Raw`
- BBH subtasks: the 24 columns whose names contain `leaderboard_bbh_`

### LiveBench (optional; multi-skill frontier)
- `tables/livebench/livebench_subcategory_results_wide.csv`
- `tables/livebench/livebench_model_metadata_numeric.csv`

## Outputs (generated artifacts)

All generated outputs are written under `outputs/` (safe to delete/regenerate) except where noted:

- `outputs/sigmoid/` and `outputs/sigmoid_size/`  
  Single-skill sigmoid frontier fits (compute axis vs model-size axis).
- `outputs/budget/`  
  Budgeted manifests + fitted curves + plots.
- `outputs/evaluation/sigmoid/`  
  Binned calibration-error evaluation (CSV summaries + heatmaps/panels).
- `outputs/sweeps_fullrange/`  
  Alpha sweeps (budget vs error curves).
- `outputs/frontier/`  
  Multi-skill frontier results (e.g., LiveBench).
- `pairwise_envelope/`  
  Pairwise envelope plots (OLL/LiveBench).
- `evaluation_pinball_baselines*/` (outside `outputs/`)  
  Pinball-loss baseline comparisons (kept separate for now).

## Repository map (main building blocks)

### Library (`skill_frontier/`)
- `skill_frontier/core/`
  - `sigmoid.py`: single-skill sigmoid frontier model + time-split definitions
  - `budget_design.py`: budget-only one-shot design (D‑optimal, I‑optimal pred‑var, balanced I‑optimal)
  - `frontier.py`: multi-skill frontier estimation (DEA/FDH/quantile support functions; smoothing)
  - `envelope.py`: monotone envelope utilities
- `skill_frontier/evaluation/`
  - `binning.py`: equal-mass binning used across evaluation/design
  - `common.py`: sigmoid predictor fit helpers
  - `metrics.py`: MAE/pinball metrics + CSV writers
- `skill_frontier/io/`
  - `csv_utils.py`: reading, compute extraction, task discovery, sanitization
  - `manifest_utils.py`: read/write manifest lists
  - `output_paths.py`: structured output-path helpers

### Scripts (`scripts/`)
- Orchestration:
  - `regenerate_all.sh`: end-to-end pipeline runner (sections: data/sigmoid/budget/eval/frontier/sweeps)
  - `regenerate_paper_artifacts_from_scratch.sh`: one-shot regeneration for paper-used artifacts only
  - `Makefile`: convenience targets (delegates to `regenerate_all.sh`)
- Single-skill frontiers:
  - `scripts/smooth_single_skill_frontier.py`: generate sigmoid frontiers (no split / year split / period4)
- Budgeted design:
  - `scripts/run/budget_only.py`: unified CLI for budget-only design + fit/plot
- Evaluation + sweeps:
  - `scripts/evaluate/sigmoid_binned_mae.py`: binned calibration-error evaluation (IS/OOS; fixed OOS bins)
  - `scripts/evaluate/sweep_alpha.py`: alpha sweeps (budgeted selection + eval + plots)
  - `scripts/evaluate/sigmoid_pinball_baselines.py`: pinball-loss baselines (sigmoid vs null/binwise/i-spline)
- Plotting:
  - `scripts/plot/eval_sigmoid.py`: heatmaps + multi-panel figures for evaluation runs
  - `scripts/plot/plot_budget_period4.py`: plots budget designs from existing manifests (period4/single_k)
  - `scripts/plot/plot_pinball_baselines.py`: pinball baseline plots

### Sampling evaluation (`eval-sampling/`)
- `eval.py`: runs **lm-eval** for a single model row from `data/top_models_by_base.csv` (Open LLM Leaderboard–style tasks by default).
- `run.sh`: Slurm job wrapper around `eval.py` for cluster runs (see below).

## `eval-sampling/run.sh`

Batch script for **one model per job**: it activates a Python environment, runs the `leaderboard` task via `eval.py`, then deletes that model’s Hugging Face Hub cache under `$HF_HOME` to free space.

**Arguments**

- First positional argument `$1` is passed to `eval.py` as `--model_id`: the **row index** in `data/top_models_by_base.csv` (same integer as in `eval.py`).

**What the job runs**

From the `eval-sampling/` directory (so `eval.py` and `data/` paths resolve), the script effectively runs:

```bash
python eval.py --tasks leaderboard --model_id "$1" --batch_size 1 --output-dir "results/${SLURM_JOB_NAME}"
```

Results are written under `$SCRATCH/control-scale/...` as implemented in `eval.py` (see that file for the exact path layout and skip-if-results-exist behavior).

**Cluster settings (edit before use)**

The script is **site-specific**: it sets `#SBATCH` resources (e.g. 2 GPUs, 12 CPUs, 128 GB, 2-day limit), loads `cuda/12.4` and `gcc`, uses a fixed conda/venv path (`source $SCRATCH/envs/control/bin/activate`), and sets `--account`. Expects `SLURM_PARTITION` to be set in the environment when you submit. Set `HF_HOME` (and typically `HF_TOKEN` if your eval needs it) before `sbatch`.

**Submit**

```bash
cd eval-sampling
export SLURM_PARTITION=your_partition
sbatch run.sh <model_id>
```

## Methods implemented (what we compute)

### 1) Single-skill sigmoid frontier fitting
- Fits a sigmoid-shaped frontier for a task score vs `log10(x)` where `x` is:
  - compute proxy (default), or
  - model size if `--size_col` is provided (producing the `*_size` outputs).
- Supports splits:
  - year split: train `<2025`, test `==2025`
  - period4: four temporal periods with `cumulative` and `single_k` train modes
- Entry point: `scripts/smooth_single_skill_frontier.py`

### 2) Coverage evaluation (binned MAE)
- Builds equal-mass bins on train `z=log10(x)` and computes per-bin coverage error `|hat_tau − tau|`.
- Reports macro/micro MAE for IS and OOS; supports fixed OOS bins (`--oos_bins test_fixed`) for stable comparisons.
- Entry point: `scripts/evaluate/sigmoid_binned_mae.py`

### 3) Budgeted one-shot design (model selection)
Selects a subset of models under a hard budget to maximize information about the sigmoid frontier.

Objectives in `skill_frontier/core/budget_design.py`:
- **D-optimal**: greedy gain-per-cost maximizing `log det(M)` (information matrix)
- **I-optimal predictive variance**: greedy step reduces average predictive variance over a z-grid
- **Balanced I-optimal**: adds a bin-balance term (encourages coverage across z bins) when `balance_lambda > 0`

Primary CLI: `scripts/run/budget_only.py`

### 4) Alpha sweeps (budget vs error curves)
- For each alpha in a list, builds budgeted manifests and evaluates coverage error:
  - year split (pre-2025 vs 2025)
  - period4 single_k (reports per-k and averaged-over-k)
- Outputs include:
  - `outputs/sweeps_fullrange/year_tasks_vs_alpha.(png|pdf)`
  - `outputs/sweeps_fullrange/period4_singlek_tasks_vs_alpha.(png|pdf)`
  - `outputs/sweeps_fullrange/period4_singlek_k{1,2,3}_tasks_vs_alpha.(png|pdf)`
- Entry point: `scripts/evaluate/sweep_alpha.py`

### 5) Multi-skill frontiers (task tradeoffs vs compute)
- Methods include DEA / FDH / directional local quantiles and monotone smoothing across compute.
- Entry point: `scripts/run/frontier_from_csv.py`
- Method details: `methods.md`

## Reproducing results

### Install (minimal)
Python 3.11+ recommended.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional:
```bash
pip install pulp  # fallback LP solver for DEA if needed
```

Developer tooling:
```bash
pip install -r requirements-dev.txt
```

Reproducibility notes and manifest-diff workflow:
- `docs/reproducibility.md`
- `scripts/regression/artifact_manifest.py`

### End-to-end (recommended)
```bash
./regenerate_all.sh
```

### Paper artifacts from scratch (targeted)
Use this to regenerate the paper-used outputs only:

```bash
./regenerate_paper_artifacts_from_scratch.sh
```

Generated targets:
- `outputs/figures_main_paper`
- `outputs/sigmoid/period4/single_k/plots`
- `outputs/sigmoid/period4/single_k/new_models`
- `outputs/sigmoid/period4/single_k/new_models_p5`
- `outputs/evaluation/sigmoid/period4/single_k/no_budget/plots`
- `evaluation_pinball/period4_singlek_no_budget/plots`
- `outputs/sigmoid_size/period4/single_k/plots`
- `outputs/sweeps_fullrange`
- `outputs_old/sigmoid/period4/single_k/plots`

Common options:
```bash
./regenerate_paper_artifacts_from_scratch.sh --no-clean
./regenerate_paper_artifacts_from_scratch.sh --skip-sweeps
./regenerate_paper_artifacts_from_scratch.sh --python python3
```

### Run one section
```bash
./regenerate_all.sh --section data
./regenerate_all.sh --section sigmoid
./regenerate_all.sh --section budget
./regenerate_all.sh --section eval
./regenerate_all.sh --section frontier
./regenerate_all.sh --section sweeps
```

## Exact pipeline commands (as invoked by `regenerate_all.sh`)

`./regenerate_all.sh` is just a wrapper around these Python entrypoints + flags (shown here so the pipeline is fully explicit/reproducible).

### Data (LiveBench merge)
```bash
python scripts/data/merge_livebench.py \
  --wide_csv tables/livebench/livebench_subcategory_results_wide.csv \
  --metadata_csv tables/livebench/livebench_model_metadata_numeric.csv \
  --out_csv tables/merged_livebench.csv \
  --min_models_per_task 20 \
  --min_values_per_row 5 \
  --clip01
```

### Sigmoid frontiers (OLL)
```bash
python scripts/smooth_single_skill_frontier.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --sigmoid --split_mode none --tau 0.98 \
  --out_dir outputs/sigmoid/no_split

python scripts/smooth_single_skill_frontier.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --sigmoid --split_mode year --tau 0.98 \
  --out_dir outputs/sigmoid/year_split

python scripts/smooth_single_skill_frontier.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --sigmoid --split_mode period4 --period4_train_mode cumulative --tau 0.98 \
  --out_dir outputs/sigmoid/period4/cumulative

python scripts/smooth_single_skill_frontier.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --sigmoid --split_mode period4 --period4_train_mode single_k --tau 0.98 \
  --out_dir outputs/sigmoid/period4/single_k
```

### Budgeted designs (OLL)
```bash
python scripts/run/budget_only.py pre2025_vs_2025 \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --budget_train_factor 200 --budget_val_factor 200 \
  --out_dir outputs \
  --binomial --m 100

python scripts/run/budget_only.py period4 \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --train_mode cumulative \
  --budget_train_factor 200 --budget_val_factor 200 \
  --out_base outputs \
  --binomial --m 100

python scripts/run/budget_only.py period4 \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --train_mode single_k \
  --budget_train_factor 200 --budget_val_factor 200 \
  --out_base outputs \
  --binomial --m 100

python scripts/plot/plot_budget_period4.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --tau 0.98 \
  --out_base outputs \
  --train_mode single_k
```

### Evaluation (binned MAE; OLL)
```bash
python scripts/evaluate/sigmoid_binned_mae.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --split_mode pre2025_vs_2025 \
  --tau 0.98 --bins 10 --min_bin_size 30 \
  --out_dir outputs/evaluation/sigmoid/year_split/no_budget

python scripts/evaluate/sigmoid_binned_mae.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --split_mode pre2025_vs_2025 \
  --tau 0.98 --bins 10 --min_bin_size 30 \
  --manifest_pre "outputs/budget/year_split/manifests/manifest__< 2025.txt" \
  --oos_bins test_fixed \
  --out_dir outputs/evaluation/sigmoid/year_split/budgeted

python scripts/evaluate/sigmoid_binned_mae.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --split_mode period4 --train_mode cumulative \
  --tau 0.98 --bins 10 --min_bin_size 30 \
  --out_base outputs/evaluation/sigmoid/period4/cumulative/no_budget

python scripts/evaluate/sigmoid_binned_mae.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --split_mode period4 --train_mode cumulative \
  --tau 0.98 --bins 10 --min_bin_size 30 \
  --manifest_base "outputs/budget/period4/cumulative/manifests" \
  --manifest_apply train_only \
  --oos_bins test_fixed \
  --out_base outputs/evaluation/sigmoid/period4/cumulative/budgeted

python scripts/evaluate/sigmoid_binned_mae.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --split_mode period4 --train_mode single_k \
  --tau 0.98 --bins 10 --min_bin_size 30 \
  --out_base outputs/evaluation/sigmoid/period4/single_k/no_budget

python scripts/evaluate/sigmoid_binned_mae.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --split_mode period4 --train_mode single_k \
  --tau 0.98 --bins 10 --min_bin_size 30 \
  --manifest_base "outputs/budget/period4/single_k/manifests" \
  --manifest_apply train_only \
  --oos_bins test_fixed \
  --out_base outputs/evaluation/sigmoid/period4/single_k/budgeted
```

### Evaluation plots (heatmaps + panels; period4/single_k)
These plots read the evaluation folders produced above and write figures into the corresponding `plots/` subfolders.

```bash
# No-budget (compute axis)
python scripts/plot/eval_sigmoid.py \
  --period4_singlek_base outputs/evaluation/sigmoid/period4/single_k/no_budget \
  --make_calib_summary_panels

# Budgeted (compute axis)
python scripts/plot/eval_sigmoid.py \
  --period4_singlek_base outputs/evaluation/sigmoid/period4/single_k/budgeted \
  --make_calib_summary_panels
```

### Size-axis variants (model size as x; optional)
These runs replace `x = compute` with `x = "#Params (B)"` (log-scaled), and write into the `*_size` output folders.

```bash
# Sigmoid frontiers on size axis (period4/single_k example)
python scripts/smooth_single_skill_frontier.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --size_col "#Params (B)" \
  --sigmoid --split_mode period4 --period4_train_mode single_k --tau 0.98 \
  --out_dir outputs/sigmoid_size/period4/single_k

# Evaluation on size axis (period4/single_k)
python scripts/evaluate/sigmoid_binned_mae.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --size_col "#Params (B)" \
  --split_mode period4 --train_mode single_k \
  --tau 0.98 --bins 10 --min_bin_size 30 \
  --out_base outputs/evaluation/sigmoid/period4/single_k/no_budget_size

python scripts/evaluate/sigmoid_binned_mae.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --size_col "#Params (B)" \
  --split_mode period4 --train_mode single_k \
  --tau 0.98 --bins 10 --min_bin_size 30 \
  --manifest_base "outputs/budget/period4/single_k/manifests" \
  --manifest_apply train_only \
  --oos_bins test_fixed \
  --out_base outputs/evaluation/sigmoid/period4/single_k/budgeted_size

# Plots for the size-axis evaluation runs
python scripts/plot/eval_sigmoid.py \
  --period4_singlek_base outputs/evaluation/sigmoid/period4/single_k/no_budget_size \
  --bin_x_label "Bin Upper Params (B)"
python scripts/plot/eval_sigmoid.py \
  --period4_singlek_base outputs/evaluation/sigmoid/period4/single_k/budgeted_size \
  --bin_x_label "Bin Upper Params (B)"
```

### Multi-skill frontier + envelopes (LiveBench + OLL)
```bash
python scripts/run/frontier_from_csv.py \
  --csv tables/merged_livebench.csv \
  --model_col model \
  --logC_col logC \
  --task_prefix a_ \
  --output_dir outputs/frontier/livebench \
  --split_output_dir \
  --csv_subdir csv \
  --plots_subdir plots/per_task \
  --pairwise_dir plots/pairwise \
  --num_C_grid 16 \
  --num_directions 128 \
  --write_vertices

python scripts/data/generate_envelopes.py --dataset oll \
  --oll_csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv

python scripts/data/generate_envelopes.py --dataset livebench \
  --livebench_csv tables/merged_livebench.csv
```

### Alpha sweeps (OLL)
```bash
python scripts/evaluate/sweep_alpha.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --alphas 5 10 20 50 100 \
  --design_objective i_optimal_predvar_balanced \
  --balance_lambda 0.1 \
  --exchange_passes 2 \
  --out_dir outputs/sweeps_fullrange
```

### Pinball evaluation + baselines (period4/single_k; optional)

```bash
# Pinball-loss evaluation (writes into evaluation_pinball/..., not outputs/)
python scripts/evaluate/sigmoid_binned_pinball.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --tau 0.98 --bins 10 --min_bin_size 30 \
  --out_base evaluation_pinball/period4_singlek_no_budget

# Baseline comparisons (main 6 tasks)
python scripts/evaluate/sigmoid_pinball_baselines.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --tau 0.98 --bins 10 --min_bin_size 30 \
  --task_group main \
  --out_base evaluation_pinball_baselines/period4_singlek_no_budget

# Baseline comparisons (BBH subtasks)
python scripts/evaluate/sigmoid_pinball_baselines.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --tau 0.98 --bins 10 --min_bin_size 30 \
  --task_group bbh_subtasks \
  --out_base evaluation_pinball_baselines/period4_singlek_no_budget_bbh_subtasks

# Plot baseline figures
python scripts/plot/plot_pinball_baselines.py \
  --base_dir_main evaluation_pinball_baselines/period4_singlek_no_budget \
  --base_dir_bbh  evaluation_pinball_baselines/period4_singlek_no_budget_bbh_subtasks
```

## Commands used in this session (to generate the current sweep outputs)

These are the exact commands I ran to produce/refresh the current contents under `outputs/sweeps_fullrange/` in this workspace:

### 1) Full sweep regeneration (design + eval + plots)
```bash
./regenerate_all.sh --section sweeps
```

In the current `regenerate_all.sh`, the sweep uses:
- alphas: `5 10 20 50 100`
- design objective: `i_optimal_predvar_balanced`
- `balance_lambda`: `0.1`

### 2) Plot-only refresh (no recomputation)
Used to regenerate figures (including `outputs/sweeps_fullrange/period4_singlek_tasks_vs_alpha.png`) from existing `outputs/sweeps_fullrange/eval_*` folders after plot-style changes:

```bash
python scripts/evaluate/sweep_alpha.py \
  --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
  --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
  --compute_multiplier 6.0 \
  --alphas 5 10 20 50 100 \
  --design_objective i_optimal_predvar_balanced \
  --balance_lambda 0.1 \
  --exchange_passes 2 \
  --out_dir outputs/sweeps_fullrange \
  --plot_only
```

## Pointers
- Regeneration walkthrough: `REGENERATION_GUIDE.md`
- Output layout rationale: `OUTPUT_RESTRUCTURING.md`
- Multi-skill frontier method details: `methods.md`
- Reproducibility contract: `docs/reproducibility.md`
- Engineering hardening backlog: `docs/code-hardening-plan.md`
- Contribution guide: `CONTRIBUTING.md`
- Citation metadata: `CITATION.cff`
- License: `LICENSE`

# Bibliography

```bibtex
@article{zhang2026prescriptive,
      title={Prescriptive Scaling Reveals the Evolution of Language Model Capabilities}, 
      author={Hanlin Zhang and Jikai Jin and Vasilis Syrgkanis and Sham Kakade},
      year={2026},
      eprint={2602.15327},
      archivePrefix={arXiv},
}
```