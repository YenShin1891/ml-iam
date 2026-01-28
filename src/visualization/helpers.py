# Shared plotting helpers (migrated from utils.plot_helpers)
import io, math
from typing import Callable, List, Optional, Sequence, Tuple
import logging
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image

__all__ = ['make_grid','render_external_plot','build_feature_display_names','filter_by_region','sample_scenario_groups']

# Central visualization/SHAP configuration (single source of truth)
DEFAULT_REGION: str = "R10"
SHAP_MAX_SCENARIO_GROUPS: int = 300

def make_grid(n_items: int, rows: Optional[int] = None, cols: Optional[int] = None, *, base_figsize=(20, 20)):
    if rows is None or cols is None:
        import math as _m
        side = _m.ceil(_m.sqrt(n_items))
        rows = cols = side
    fig, axes = plt.subplots(rows, cols, figsize=base_figsize)
    try:
        axes_iter = axes.flatten()  # type: ignore[attr-defined]
    except Exception:
        axes_iter = [axes]
    return fig, list(axes_iter)

def render_external_plot(ax, plot_fn: Callable[[Figure], None]):
    fig = plt.figure()
    try:
        plot_fn(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        ax.imshow(img)
        ax.axis('off')
    finally:
        plt.close(fig)

def _make_display_name(feature: str) -> str:
    import re

    missing_suffix = False
    base = feature

    # Detect and strip missingness suffix before other transforms
    if base.endswith('_is_missing'):
        missing_suffix = True
        base = base[:-len('_is_missing')]

    timestep = None
    m_timestep = re.match(r'^timestep_(\d+)_(.+)$', base)
    if m_timestep:
        timestep = int(m_timestep.group(1))
        base = m_timestep.group(2)

    lag_years = None
    m_prev = re.match(r'^prev(\d*)_(.+)$', base)
    if m_prev:
        lag_str, base = m_prev.group(1), m_prev.group(2)
        lag_years = 5 if lag_str == '' else int(lag_str) * 5

    if lag_years is not None:
        descriptor = f" (last {lag_years}y)"
    elif timestep is not None:
        descriptor = " (current)" if timestep == 0 else f" (last {timestep * 5}y)"
    else:
        descriptor = " (current)"

    display = f"{base}{descriptor}"
    if missing_suffix:
        display = f"{display} N/A"
    return display

def build_feature_display_names(features: Sequence[str]) -> List[str]:
    return [_make_display_name(f) for f in features]


def draw_shap_beeswarm(
    ax,
    shap_matrix,
    X_matrix,
    feature_display_names: Sequence[str],
    *,
    max_display: int = 8,
    xlim_range: Optional[tuple] = None,
    point_size: float = 5.0,
):
    """Draw SHAP beeswarm (dot) summary plot with color scale.

    Parameters
    - ax: Matplotlib Axes to draw the rendered image into via render_external_plot caller.
    - shap_matrix: array-like (n_samples, n_features)
    - X_matrix: array-like (n_samples, n_features) matching shap_matrix
    - feature_display_names: names for features, length = n_features
    - max_display: top features to display
    - xlim_range: optional (min, max) for x-axis
    """
    import numpy as _np
    import shap as _shap
    import matplotlib.pyplot as _plt

    shap_arr = _np.asarray(shap_matrix)
    X_arr = _np.asarray(X_matrix)
    if shap_arr.shape != X_arr.shape:
        raise ValueError("shap_matrix and X_matrix must have the same shape")
    if len(feature_display_names) != shap_arr.shape[1]:
        raise ValueError("feature_display_names length must match number of columns")

    # Ensure SHAP draws into the provided Axes rather than whatever
    # happens to be current in the global pyplot state.
    try:
        _plt.sca(ax)
    except Exception:
        # If we can't set the current axes, fall back to SHAP's default
        # behaviour (which will use the current pyplot axes).
        pass

    _shap.summary_plot(
        shap_arr,
        X_arr,
        feature_names=list(feature_display_names),
        max_display=max_display,
        plot_type='dot',  # beeswarm
        show=False,
    )
    # Reduce dot size by adjusting PathCollections on the target axes
    try:
        for coll in getattr(ax, 'collections', []):
            try:
                sizes = coll.get_sizes()
                if sizes is not None and len(sizes) > 0:
                    coll.set_sizes(_np.full_like(sizes, point_size, dtype=float))
                else:
                    coll.set_sizes([point_size])
            except Exception:
                continue
    except Exception:
        pass
    if xlim_range is not None:
        try:
            ax.set_xlim(xlim_range[0], xlim_range[1])
        except Exception:
            _plt.xlim(xlim_range[0], xlim_range[1])


def filter_by_region(
    df,
    region: Optional[str],
    *,
    log_prefix: str = "Applied region filter",
    mode: str = "exact",
):
    """Filter a DataFrame by Region with normalization and graceful fallback.

    Parameters
    ----------
    df : DataFrame
        Input data with a 'Region' column.
    region : str or None
        Region label or prefix to match.
    log_prefix : str
        Prefix for log messages.
    mode : {"exact","prefix"}
        - "exact": case-insensitive equality match (with fallback to contains).
        - "prefix": case-insensitive "starts with" match (no contains fallback).

    Returns (filtered_df, pre_rows, post_rows, matched_values).
    """
    try:
        import pandas as _pd  # local import to avoid hard dep at module import time
    except Exception:  # pragma: no cover
        # If pandas missing, just return unchanged
        return df, len(df) if hasattr(df, '__len__') else 0, len(df) if hasattr(df, '__len__') else 0, []

    if region is None or not isinstance(df, _pd.DataFrame) or 'Region' not in df.columns:
        return df, len(df), len(df), []

    pre_rows = len(df)
    s = df['Region'].astype(str).str.strip()
    s_lower = s.str.lower()
    target = str(region).strip().lower()

    import logging as _logging

    if mode == "prefix":
        # Case-insensitive "starts with" match, aligned with dashboard region bucketing
        prefix_mask = s_lower.str.startswith(target)
        if prefix_mask.any():
            filtered = df[prefix_mask]
            matched = sorted(s[prefix_mask].unique().tolist())
            _logging.info(
                f"{log_prefix} prefix '{region}': {pre_rows} -> {len(filtered)} rows (prefix matches: {matched})"
            )
            return filtered, pre_rows, len(filtered), matched
        _logging.warning(
            f"{log_prefix} prefix '{region}': 0 matches in Region column; proceeding without filter"
        )
        return df, pre_rows, pre_rows, []

    # Default: exact-then-contains behaviour
    exact_mask = (s_lower == target)
    if exact_mask.any():
        filtered = df[exact_mask]
        matched = sorted(s[exact_mask].unique().tolist())
        _logging.info(f"{log_prefix} '{region}': {pre_rows} -> {len(filtered)} rows (exact: {matched})")
        return filtered, pre_rows, len(filtered), matched

    import re as _re
    contains_mask = s_lower.str.contains(_re.escape(target), na=False)
    if contains_mask.any():
        filtered = df[contains_mask]
        matched = sorted(s[contains_mask].unique().tolist())
        _logging.info(f"{log_prefix} '{region}': {pre_rows} -> {len(filtered)} rows (contains: {matched})")
        return filtered, pre_rows, len(filtered), matched

    _logging.warning(f"{log_prefix} '{region}': 0 matches in Region column; proceeding without filter")
    return df, pre_rows, pre_rows, []


def sample_scenario_groups(
    df,
    *,
    group_cols: Sequence[str] = ("Model", "Scenario"),
    max_groups: int = SHAP_MAX_SCENARIO_GROUPS,
    log_prefix: str = "SHAP scenario sampling",
) -> Tuple[object, int, int, List[str]]:
    """Sample complete scenario groups for SHAP while logging totals.

    Returns (group_keys_df, total_groups, used_groups, group_col_list) where
    group_keys_df has one row per selected group and only the grouping columns.
    If none of the group columns are present, an empty DataFrame is returned
    and total/used groups are reported as 0.
    """
    try:
        import pandas as _pd
    except Exception:  # pragma: no cover
        return df, 0, 0, []

    if not isinstance(df, _pd.DataFrame):
        return _pd.DataFrame(), 0, 0, []

    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        logging.info(f"{log_prefix}: no scenario grouping columns found; using row-based sample only")
        return _pd.DataFrame(), 0, 0, []

    group_sizes = df.groupby(cols).size()
    total_groups = int(len(group_sizes))
    if total_groups == 0:
        logging.info(f"{log_prefix}: 0 scenario groups found (by {','.join(cols)})")
        return _pd.DataFrame(), 0, 0, list(cols)

    max_sequences = min(max_groups, total_groups)

    if total_groups > max_sequences:
        import numpy as _np
        sampled_idx = _np.random.choice(total_groups, max_sequences, replace=False)
        selected_group_keys = group_sizes.iloc[sampled_idx].index
        # Build a DataFrame of unique selected group keys for joining
        group_df = _pd.DataFrame(list(selected_group_keys), columns=cols)
        logging.info(
            "%s: sampled %d of %d scenario groups (by %s)",
            log_prefix,
            max_sequences,
            total_groups,
            ",".join(cols),
        )
        used_groups = max_sequences
    else:
        # All groups are kept; return all unique keys
        group_df = df[list(cols)].drop_duplicates().reset_index(drop=True)
        logging.info(
            "%s: using all %d scenario groups (by %s)",
            log_prefix,
            total_groups,
            ",".join(cols),
        )
        used_groups = total_groups

    return group_df, total_groups, used_groups, list(cols)

