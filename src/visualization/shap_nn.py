# Neural network SHAP plotting (migrated from utils.plot_shap_nn)
import os, logging, numpy as np, pandas as pd, shap, torch
from typing import List, Optional, Dict
from configs.paths import RESULTS_PATH
from configs.data import NON_FEATURE_COLUMNS, OUTPUT_UNITS
from .helpers import make_grid, render_external_plot, build_feature_display_names, sequence_time_labels

__all__ = [
    'get_lstm_shap_values','plot_lstm_shap','draw_lstm_all_timesteps_shap_plot','draw_temporal_shap_plot','create_timestep_comparison_plots'
]

def _to_numpy(x):
    try:
        if hasattr(x, 'detach') and callable(getattr(x, 'detach')):
            return x.detach().cpu().numpy()
        if hasattr(x, 'cpu') and callable(getattr(x, 'cpu')) and hasattr(x, 'numpy'):
            return x.cpu().numpy()
        if hasattr(x, 'numpy') and callable(getattr(x, 'numpy')):
            return x.numpy()
    except Exception:
        pass
    import numpy as _np
    return _np.array(x)

def get_lstm_shap_values(run_id, X_test: pd.DataFrame, sequence_length=1):
    from src.trainers.lstm_trainer import LSTMModel
    logging.info("Loading LSTM model...")
    model_path = os.path.join(RESULTS_PATH, run_id, "final", "best.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model checkpoint not found: {model_path}")
    model = LSTMModel.load_from_checkpoint(model_path)
    model.eval()
    from src.utils.utils import load_session_state
    session_state = load_session_state(run_id)
    scaler_X = session_state.get("lstm_scaler_X")
    features = session_state.get("features")
    targets = session_state.get("targets")
    if scaler_X is None:
        raise ValueError("LSTM scaler_X not found in session state")
    def preprocess_features(data, features, scaler_X, mask_value=-1.0):
        X = data[features].copy()
        from configs.data import CATEGORICAL_COLUMNS
        cat_cols = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes
        X_filled = X.fillna(mask_value).astype(np.float32)
        return scaler_X.transform(X_filled)
    def create_sequences(X_scaled, seq_len):
        import torch as _torch
        seqs = []
        for i in range(len(X_scaled) - seq_len + 1):
            seqs.append(_torch.FloatTensor(X_scaled[i:i+seq_len]))
        return _torch.stack(seqs) if seqs else _torch.empty(0, seq_len, X_scaled.shape[1])
    background_size = min(200, len(X_test))
    background_scaled = preprocess_features(X_test.iloc[:background_size], features, scaler_X)
    background_sequences = create_sequences(background_scaled, sequence_length)
    if len(background_sequences) > 50:
        import numpy as _np
        idx = _np.random.choice(len(background_sequences), 50, replace=False)
        background_data = background_sequences[idx]
    else:
        background_data = background_sequences
    test_size = min(100, len(X_test))
    test_scaled = preprocess_features(X_test.iloc[:test_size], features, scaler_X)
    test_inputs = create_sequences(test_scaled, sequence_length)
    import torch as _torch
    class LSTMWrapperForSHAP(_torch.nn.Module):
        def __init__(self, lstm_model, seq_len):
            super().__init__()
            self.lstm_model = lstm_model
            self.seq_len = seq_len
        def forward(self, x):
            batch_size = x.shape[0]
            mask = _torch.ones(batch_size, self.seq_len, dtype=_torch.float32, device=x.device)
            return self.lstm_model(x, mask=mask, teacher_forcing=False)
    wrapper = LSTMWrapperForSHAP(model, sequence_length)
    wrapper.eval()
    background_data.requires_grad_(True)
    test_inputs.requires_grad_(True)
    explainer = shap.DeepExplainer(wrapper, background_data)
    logging.info("Calculating LSTM SHAP values...")
    shap_values = explainer.shap_values(test_inputs, check_additivity=False)
    import numpy as _np
    if isinstance(shap_values, list):
        shap_values = [_to_numpy(sv) for sv in shap_values]
        shap_values = _np.array(shap_values, dtype=_np.float64)
        shap_values = _np.transpose(shap_values, (1, 2, 3, 0))
    else:
        shap_values = _to_numpy(shap_values)
        shap_values = _np.array(shap_values, dtype=_np.float64)
        if shap_values.ndim == 3:
            shap_values = _np.expand_dims(shap_values, axis=-1)
    original_temporal_shap = shap_values.copy()
    averaged = _np.mean(shap_values, axis=1)
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots"), exist_ok=True)
    _np.save(os.path.join(RESULTS_PATH, run_id, "plots", "lstm_shap_values_temporal.npy"), original_temporal_shap)
    _np.save(os.path.join(RESULTS_PATH, run_id, "plots", "lstm_shap_values.npy"), averaged)
    n_samples = averaged.shape[0]
    X_processed = X_test.iloc[:min(test_size, n_samples)][features]
    test_sequences_np = _to_numpy(test_inputs)
    return original_temporal_shap, averaged, X_processed, test_sequences_np

def plot_lstm_shap(run_id, X_test_with_index: pd.DataFrame, features: List[str], targets: List[str], sequence_length=1, feature_name_map: Optional[Dict[str, str]] = None):
    logging.info("Creating LSTM SHAP plots...")
    model_path = os.path.join(RESULTS_PATH, run_id, "final", "best.ckpt")
    if not os.path.exists(model_path):
        logging.warning("Skipping LSTM SHAP plots: model checkpoint not found at %s", model_path)
        return
    X_test = X_test_with_index.drop(columns=NON_FEATURE_COLUMNS, errors="ignore" ).reset_index(drop=True)
    import numpy as _np
    if X_test.shape[0] > 100:
        idx = _np.random.choice(X_test.shape[0], 100, replace=False)
        X_test = X_test.iloc[idx]
    try:
        temporal_shap, averaged, X_proc, test_seq = get_lstm_shap_values(run_id, X_test, sequence_length)
        draw_lstm_all_timesteps_shap_plot(run_id, temporal_shap, test_seq, features, targets, sequence_length, feature_name_map)
        draw_temporal_shap_plot(run_id, temporal_shap, X_proc, features, targets, sequence_length, feature_name_map)
    except Exception as e:
        logging.error("Failed to create LSTM SHAP plots: %s", e)
        logging.exception("Full error traceback:")

def draw_lstm_all_timesteps_shap_plot(run_id: str, temporal_shap_values, test_sequences_np, features: List[str], targets: List[str], sequence_length: int, feature_name_map: Optional[Dict[str, str]] = None) -> None:
    import numpy as _np
    if sequence_length <= 0:
        logging.warning("Invalid sequence_length=%d; skipping all-timesteps SHAP plot", sequence_length)
        return
    x_flat_parts = [test_sequences_np[:, t, :] for t in range(sequence_length)]
    X_flat = _np.concatenate(x_flat_parts, axis=1)
    feature_name_map = feature_name_map or {}
    base_names = [feature_name_map.get(f, f) for f in features]
    labels = sequence_time_labels(sequence_length)
    display_names = [f"{bn} ({labels[t]})" for t in range(sequence_length) for bn in base_names]
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    num_targets = len(targets)
    fig, axes = make_grid(num_targets, base_figsize=(20, 20))
    for i, ax in enumerate(axes):
        if i >= num_targets:
            ax.axis('off')
            continue
        def _plot(fig_local):
            shap_flat_parts = [temporal_shap_values[:, t, :, i] for t in range(sequence_length)]
            shap_flat = _np.concatenate(shap_flat_parts, axis=1)
            shap.summary_plot(
                shap_flat,
                X_flat,
                feature_names=display_names,
                max_display=8,
                plot_type='violin',
                show=False,
            )
            fig_local.tight_layout()
        render_external_plot(ax, _plot)
        ax.set_title(f"SHAP across all timesteps: {targets[i]} ({OUTPUT_UNITS[i]})")
    fig.tight_layout()
    os.makedirs(os.path.join(RESULTS_PATH, run_id, 'plots'), exist_ok=True)
    fig.savefig(os.path.join(RESULTS_PATH, run_id, 'plots', 'lstm_shap_plot.png'))
    plt.close(fig)

def draw_temporal_shap_plot(run_id: str, temporal_shap_values, X_test: pd.DataFrame, features: List[str], targets: List[str], sequence_length: int, feature_name_map: Optional[Dict[str, str]] = None) -> None:
    import matplotlib.pyplot as plt, numpy as _np
    from tqdm import tqdm
    plt.rcParams.update({'font.size': 12})
    num_targets = len(targets)
    fig, axes = make_grid(num_targets, base_figsize=(20, 20))
    for i, ax in tqdm(enumerate(axes), total=num_targets, desc="Creating temporal SHAP plots"):
        if i >= num_targets:
            ax.axis('off')
            continue
        target_shap = temporal_shap_values[:, :, :, i]
        avg_importance = _np.mean(_np.abs(target_shap), axis=(0, 1))
        top_idx = _np.argsort(avg_importance)[-8:][::-1]
        time_importance = _np.mean(_np.abs(target_shap[:, :, top_idx]), axis=0)
        im = ax.imshow(time_importance.T, aspect='auto', cmap='viridis', interpolation='nearest')
        labels = sequence_time_labels(sequence_length)
        ax.set_xticks(range(sequence_length))
        ax.set_xticklabels(labels)
        display_names = build_feature_display_names([features[idx] for idx in top_idx], feature_name_map)
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([n[:30] + '...' if len(n) > 30 else n for n in display_names], fontsize=10)
        ax.set_title(f"Temporal SHAP: {targets[i]} ({OUTPUT_UNITS[i]})", fontsize=14)
        ax.set_xlabel("Time Step in Sequence", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    os.makedirs(os.path.join(RESULTS_PATH, run_id, 'plots'), exist_ok=True)
    fig.savefig(os.path.join(RESULTS_PATH, run_id, 'plots', 'lstm_temporal_shap_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    create_timestep_comparison_plots(run_id, temporal_shap_values, features, targets, sequence_length, feature_name_map)

def create_timestep_comparison_plots(run_id: str, temporal_shap_values, features: List[str], targets: List[str], sequence_length: int, feature_name_map: Optional[Dict[str, str]] = None) -> None:
    import matplotlib.pyplot as plt, numpy as _np
    from matplotlib import cm
    plt.rcParams.update({'font.size': 12})
    fig, axes = make_grid(len(targets), base_figsize=(20, 20))
    for i, ax in enumerate(axes):
        if i >= len(targets):
            ax.axis('off')
            continue
        target_shap = temporal_shap_values[:, :, :, i]
        timestep_importance = _np.mean(_np.abs(target_shap), axis=(0, 2))
        labels = sequence_time_labels(sequence_length)
        colors = cm.get_cmap('viridis')(_np.linspace(0, 1, sequence_length))
        bars = ax.bar(range(sequence_length), timestep_importance, color=colors, alpha=0.8)
        ax.set_title(f"Time Step Importance: {targets[i]} ({OUTPUT_UNITS[i]})", fontsize=14)
        ax.set_xlabel("Time Step in Sequence", fontsize=12)
        ax.set_ylabel("Average |SHAP| Value", fontsize=12)
        ax.set_xticks(range(sequence_length))
        ax.set_xticklabels(labels)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + h*0.01, f"{h:.3f}", ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_PATH, run_id, 'plots', 'lstm_timestep_importance.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info("Temporal SHAP plots saved")
