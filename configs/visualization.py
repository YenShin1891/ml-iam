"""
Centralized configuration for visualization.
Keep it lightweight for easy imports.
"""

from __future__ import annotations

from typing import Tuple

# -----------------------------
# Region filtering defaults
# -----------------------------

# Default region used by SHAP plotting when a caller does not specify one.
# Kept as-is to preserve existing behavior.
DEFAULT_REGION: str = "R10"


# -----------------------------
# SHAP sampling defaults
# -----------------------------

# Upper bound on the number of unique (Model, Scenario) groups to include
# when creating SHAP plots. This keeps SHAP computation tractable.
SHAP_MAX_SCENARIO_GROUPS: int = 300

# Number of interpolation points expected gradients evaluates per explained
# window for the neural-network models: this many pairs of a background
# window and a position between it and the explained one.  Higher is more
# accurate and proportionally slower.  The estimate is unbiased and its noise
# falls with this number; at 400 the attributions of a window miss the
# prediction difference they decompose by roughly 6-7% for both models.
SHAP_GRADIENT_NSAMPLES: int = 400

# How many of those points go through the model at once.  It does not change
# the result, only the time and memory: the TFT needs about 3 GB per 100.
SHAP_GRADIENT_BATCH_SIZE: int = 200

# How many sequence windows the neural-network models explain, and how many
# they use as the expected-gradients background.  Both are seeded random
# draws from every window of the sampled scenario groups.
SHAP_NN_EXPLAINED_WINDOWS: int = 500
SHAP_NN_BACKGROUND_WINDOWS: int = 200

# The x axis of every SHAP beeswarm.  Each output's SHAP values are divided
# by its sum over features of mean |SHAP|, so a value is measured against
# that output's total attribution whatever units the model predicts in --
# XGBoost's scaled targets, the LSTM's standardised ones or the TFT's native
# ones.
SHAP_NORMALISED_XLABEL: str = "Normalized SHAP value\n(SHAP / the output's total mean |SHAP|)"


# -----------------------------
# SHAP plot rendering defaults
# -----------------------------

# Grid figure size for multi-target SHAP summary plots.
SHAP_GRID_FIGSIZE: Tuple[int, int] = (20, 20)

# Individual target SHAP figure size.
SHAP_INDIVIDUAL_FIGSIZE: Tuple[int, int] = (10, 8)

# Font size for SHAP summary plots.
SHAP_FONT_SIZE: int = 12

# Number of features to display in SHAP beeswarm.
SHAP_MAX_DISPLAY: int = 8
SHAP_MAX_DISPLAY_EXCLUDE_TOP: int = 7


# -----------------------------
# Trajectory / scatter defaults
# -----------------------------

# Global matplotlib font size used by trajectory/scatter multi-panel figures.
PLOT_FONT_SIZE: int = 16

# Axis/title sizing used across scatter/trajectory plots.
AXIS_LABEL_FONTSIZE: int = 19
TICK_LABELSIZE: int = 15
LEGEND_FONTSIZE: int = 11
R2_ANNOTATION_FONTSIZE: int = 17

# Grid layouts (assumes up to 9 targets).
PLOT_GRID_ROWS: int = 3
PLOT_GRID_COLS: int = 3

# Figure sizes
SCATTER_GRID_FIGSIZE: Tuple[int, int] = (20, 20)
SCATTER_INDIVIDUAL_FIGSIZE: Tuple[int, int] = (7, 7)
TRAJECTORY_GRID_FIGSIZE: Tuple[int, int] = (15, 15)
TRAJECTORY_INDIVIDUAL_FIGSIZE: Tuple[int, int] = (6, 6)

# Axis tick locator defaults
AXIS_NBINS: int = 5
Y_AXIS_NBINS_TRAJECTORY: int = 6
