from __future__ import annotations

# Axis label strings shared across plotting scripts.

# NOTE: This label is intentionally cased to match legacy figures and is used as
# a trigger for special tick formatting (see `skill_frontier.plotting.axis_formatting`).
PRETRAINING_COMPUTE_FLOPS_LABEL = "Pretraining Compute (FLOPs)"

# Model size is typically stored in billions of parameters (#Params (B)) in CSVs, but
# we format ticks as absolute parameter counts (multiply by 1e9) to match the
# compute-axis plotting convention.
MODEL_SIZE_PARAMS_LABEL = "Model Size (#Params)"

# Evaluation heatmaps typically bin compute in units of 1e21 FLOPs.
BIN_UPPER_FLOPS_1E21_LABEL = r"Bin Upper FLOPs $(\times 10^{21})$"
