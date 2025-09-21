# Shared plotting helpers (migrated from utils.plot_helpers)
import io, math
from typing import Callable, List, Optional, Sequence
import matplotlib.pyplot as plt
from PIL import Image

__all__ = ['make_grid','render_external_plot','build_feature_display_names','sequence_time_labels']

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
    m = re.match(r'^prev(\d*)_(.+)$', feature)
    if m:
        n_str, base = m.group(1), m.group(2)
        years = 5 if n_str == '' else int(n_str) * 5
        return f"{base} (last {years}y)"
    return f"{feature} (current)"

def build_feature_display_names(features: Sequence[str], name_map: Optional[dict] = None) -> List[str]:
    name_map = name_map or {}
    return [name_map.get(f, _make_display_name(f)) for f in features]

def sequence_time_labels(sequence_length: int) -> List[str]:
    labels = []
    for t in range(sequence_length):
        lag = sequence_length - 1 - t
        labels.append('current' if lag == 0 else f'last {lag*5}y')
    return labels
