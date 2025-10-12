# Shared plotting helpers (migrated from utils.plot_helpers)
import io, math
from typing import Callable, List, Optional, Sequence
import matplotlib.pyplot as plt
from PIL import Image

__all__ = ['make_grid','render_external_plot','build_feature_display_names']

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

def render_external_plot(ax, plot_fn: Callable[[plt.Figure], None]):
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

