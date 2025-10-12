"""Visualization API.

Import from this module for stable plotting & SHAP functions.

Torch / neural SHAP is optional: if PyTorch is not installed you can
still use trajectory and XGBoost SHAP utilities without importing torch.
"""
from .trajectories import (
    preprocess_data,
    format_large_numbers,
    create_single_trajectory_plot,
    configure_axes,
    plot_scatter,
    plot_trajectories,
    get_saved_plots_metadata,
)
from .shap_xgb import (
    get_shap_values,
    transform_outputs_to_former_inputs,
    draw_shap_plot,
    plot_xgb_shap,
)
from .helpers import (
    make_grid,
    render_external_plot,
    build_feature_display_names,
)


_nn_exports = [
    'get_lstm_shap_values',
    'plot_lstm_shap',
    'draw_lstm_all_timesteps_shap_plot',
    'draw_temporal_shap_plot',
    'create_timestep_comparison_plots',
    'get_tft_shap_values',
    'plot_tft_shap',
    'draw_shap_all_timesteps_plot',
    'get_shap_values',
    'plot_nn_shap',
]

try:  # Optional neural SHAP (requires torch)
    from .shap_nn import (  # type: ignore
        get_lstm_shap_values,
        plot_lstm_shap,
        draw_lstm_all_timesteps_shap_plot,
        draw_temporal_shap_plot,
        create_timestep_comparison_plots,
        get_tft_shap_values,
        plot_tft_shap,
        draw_shap_all_timesteps_plot,
        get_shap_values,
        plot_nn_shap,
    )
    _HAS_NN_SHAP = True
except ModuleNotFoundError as e:
    # If torch (or another nn dependency) is missing provide stubs so import still succeeds.
    _HAS_NN_SHAP = False
    def _nn_unavailable(*args, **kwargs):  # noqa: D401
        raise RuntimeError("Neural SHAP utilities unavailable: optional dependency missing (likely 'torch'). "
                           "Install torch to enable LSTM / sequence SHAP plotting.")
    # Create stub symbols
    get_lstm_shap_values = plot_lstm_shap = \
        draw_lstm_all_timesteps_shap_plot = draw_temporal_shap_plot = \
        create_timestep_comparison_plots = get_tft_shap_values = \
        plot_tft_shap = draw_shap_all_timesteps_plot = \
        get_shap_values = plot_nn_shap = _nn_unavailable
        
__all__ = [
    # Trajectories
    'preprocess_data','format_large_numbers','create_single_trajectory_plot','configure_axes',
    'plot_scatter','plot_trajectories','get_saved_plots_metadata',
    # XGB SHAP
    'get_shap_values','transform_outputs_to_former_inputs','draw_shap_plot','plot_xgb_shap',
    # Helpers
    'make_grid','render_external_plot','build_feature_display_names'
]

# Always expose NN names (either real or stubs) so wildcard imports work.
__all__.extend(_nn_exports)
