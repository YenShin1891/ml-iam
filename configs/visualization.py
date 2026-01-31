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
