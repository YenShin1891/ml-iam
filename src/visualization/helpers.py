# Shared plotting helpers (migrated from utils.plot_helpers)
import io, math
from typing import Callable, List, Optional, Sequence, Tuple
import logging
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
import pandas as pd

from configs.visualization import DEFAULT_REGION, SHAP_MAX_SCENARIO_GROUPS, SHAP_GRID_FIGSIZE

__all__ = [
    'make_grid',
    'render_external_plot',
    'build_feature_display_names',
    'filter_by_region',
    'filter_index_frame_by_region',
    'sample_scenario_groups',
]

def make_grid(
    n_items: int,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    *,
    base_figsize=SHAP_GRID_FIGSIZE,
):
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


def filter_index_frame_by_region(
    df: pd.DataFrame,
    region: Optional[str],
    *,
    default_region: str = DEFAULT_REGION,
    region_series: Optional[pd.Series] = None,
    log_prefix: str = "Applied region filter",
) -> Tuple[pd.DataFrame, Optional["np.ndarray"], int, int, List[str], str]:
    """Filter an index-frame by region with consistent prefix/exact semantics.

    This wraps `filter_by_region` but adds two conveniences:
    - Automatically chooses `mode` ("prefix" vs "exact") using the same heuristic
      used in `shap_nn.py`.
    - Optionally filters using an external `region_series` aligned to rows (useful
      when the raw Region labels are not present in `df`).

    Returns
    -------
    filtered_df : pd.DataFrame
        Always returned with a reset integer index.
    idx : np.ndarray | None
        Positional indices (relative to the reset-index frame) used for filtering.
        None means no filter was applied (or no matches).
    pre_rows, post_rows : int
    matched_values : list[str]
    mode : str
    """
    try:
        import numpy as np  # local import to keep module lightweight
    except Exception:  # pragma: no cover
        np = None  # type: ignore[assignment]

    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(), None, 0, 0, [], "exact"

    df_aligned = df.reset_index(drop=True)
    pre_rows = len(df_aligned)
    mode = "prefix" if isinstance(region, str) and region.startswith(str(default_region)) else "exact"

    if region is None or pre_rows == 0:
        return df_aligned, None, pre_rows, pre_rows, [], mode

    if region_series is not None:
        series_aligned = region_series.reset_index(drop=True).astype(str)
        region_frame = pd.DataFrame({"Region": series_aligned})
        filtered_region_frame, _, _, matched = filter_by_region(
            region_frame,
            region,
            log_prefix=log_prefix,
            mode=mode,
        )
        if matched and np is not None:
            idx = filtered_region_frame.index.to_numpy()
            filtered_df = df_aligned.iloc[idx].reset_index(drop=True)
            return filtered_df, idx, pre_rows, len(filtered_df), matched, mode
        return df_aligned, None, pre_rows, pre_rows, [], mode

    # Filter directly on the aligned df's Region column (if present)
    filtered_df, _, _, matched = filter_by_region(
        df_aligned,
        region,
        log_prefix=log_prefix,
        mode=mode,
    )
    if matched and np is not None:
        idx = filtered_df.index.to_numpy()
        # Rebuild from df_aligned to guarantee the returned frame is aligned and clean.
        filtered_df = df_aligned.iloc[idx].reset_index(drop=True)
        return filtered_df, idx, pre_rows, len(filtered_df), matched, mode
    return df_aligned, None, pre_rows, pre_rows, [], mode


def sample_scenario_groups(
    df,
    *,
    group_cols: Sequence[str] = ("Model", "Scenario"),
    max_groups: int = SHAP_MAX_SCENARIO_GROUPS,
    log_prefix: str = "SHAP scenario sampling",
) -> Tuple[pd.DataFrame, int, int, List[str]]:
    """Sample complete scenario groups for SHAP while logging totals.

    Returns (group_keys_df, total_groups, used_groups, group_col_list) where
    group_keys_df has one row per selected group and only the grouping columns.
    If none of the group columns are present, an empty DataFrame is returned
    and total/used groups are reported as 0.
    """
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(), 0, 0, []

    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        logging.info(f"{log_prefix}: no scenario grouping columns found; using row-based sample only")
        return pd.DataFrame(), 0, 0, []

    group_sizes = df.groupby(cols).size()
    total_groups = int(len(group_sizes))
    if total_groups == 0:
        logging.info(f"{log_prefix}: 0 scenario groups found (by {','.join(cols)})")
        return pd.DataFrame(), 0, 0, list(cols)

    max_sequences = min(max_groups, total_groups)

    if total_groups > max_sequences:
        import numpy as _np
        sampled_idx = _np.random.choice(total_groups, max_sequences, replace=False)
        selected_group_keys = group_sizes.iloc[sampled_idx].index
        # Build a DataFrame of unique selected group keys for joining
        group_df = pd.DataFrame(list(selected_group_keys), columns=cols)
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

