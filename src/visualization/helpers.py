# Shared plotting helpers (migrated from utils.plot_helpers)
import io, math
from typing import Callable, List, Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image

__all__ = ['make_grid','render_external_plot','build_feature_display_names','filter_by_region']

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
    # Handle timestep prefixes for temporal models (TFT, LSTM)
    m_timestep = re.match(r'^timestep_(\d+)_(.+)$', feature)
    if m_timestep:
        t_str, base = m_timestep.group(1), m_timestep.group(2)
        t = int(t_str)

        # Handle '_is_missing' features - these are static, no temporal factor
        m_missing = re.match(r'^(.*)_is_missing$', base)
        if m_missing:
            base_feature = m_missing.group(1)
            return f"{base_feature} N/A"

        # Regular features with timestep mapping: t-0=current, t-1=(last 5y), t-2=(last 10y), etc.
        if t == 0:
            return f"{base} (current)"
        else:
            years = t * 5
            return f"{base} (last {years}y)"

    # Handle features ending with '_is_missing' (non-timestep)
    m_missing = re.match(r'^(.*)_is_missing$', feature)
    if m_missing:
        base = m_missing.group(1)
        return f"{base} N/A"

    # Handle previous year features (existing logic)
    m = re.match(r'^prev(\d*)_(.+)$', feature)
    if m:
        n_str, base = m.group(1), m.group(2)
        years = 5 if n_str == '' else int(n_str) * 5
        return f"{base} (last {years}y)"

    return f"{feature} (current)"

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

    _shap.summary_plot(
        shap_arr,
        X_arr,
        feature_names=list(feature_display_names),
        max_display=max_display,
        plot_type='dot',  # beeswarm
        show=False,
    )
    # Reduce dot size by adjusting PathCollections on current axes
    try:
        ax_obj = _plt.gca()
        for coll in getattr(ax_obj, 'collections', []):
            try:
                sizes = coll.get_sizes()
                if sizes is not None and len(sizes) > 0:
                    import numpy as _np
                    coll.set_sizes(_np.full_like(sizes, point_size, dtype=float))
                else:
                    coll.set_sizes([point_size])
            except Exception:
                continue
    except Exception:
        pass
    if xlim_range is not None:
        _plt.xlim(xlim_range[0], xlim_range[1])


def filter_by_region(
    df,
    region: Optional[str],
    *,
    log_prefix: str = "Applied region filter",
):
    """Filter a DataFrame by Region with normalization and graceful fallback.

    - Trims whitespace and compares case-insensitively.
    - If no exact matches, tries substring contains.
    - If still no matches, returns the original DataFrame unchanged.

    Returns (filtered_df, pre_rows, post_rows, matched_values)
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

    exact_mask = (s_lower == target)
    if exact_mask.any():
        filtered = df[exact_mask]
        matched = sorted(s[exact_mask].unique().tolist())
        import logging as _logging
        _logging.info(f"{log_prefix} '{region}': {pre_rows} -> {len(filtered)} rows (exact: {matched})")
        return filtered, pre_rows, len(filtered), matched

    import re as _re
    contains_mask = s_lower.str.contains(_re.escape(target), na=False)
    if contains_mask.any():
        filtered = df[contains_mask]
        matched = sorted(s[contains_mask].unique().tolist())
        import logging as _logging
        _logging.info(f"{log_prefix} '{region}': {pre_rows} -> {len(filtered)} rows (contains: {matched})")
        return filtered, pre_rows, len(filtered), matched

    import logging as _logging
    _logging.warning(f"{log_prefix} '{region}': 0 matches in Region column; proceeding without filter")
    return df, pre_rows, pre_rows, []

