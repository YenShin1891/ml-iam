# Neural network SHAP plotting (migrated from utils.plot_shap_nn)
import os, logging, numpy as np, pandas as pd, shap, torch
from typing import List, Optional, Dict
from configs.paths import RESULTS_PATH
from configs.data import NON_FEATURE_COLUMNS, OUTPUT_UNITS
from .helpers import make_grid, render_external_plot, build_feature_display_names

__all__ = [
    'get_lstm_shap_values','plot_lstm_shap','draw_lstm_all_timesteps_shap_plot','draw_temporal_shap_plot','create_timestep_comparison_plots',
    'get_tft_shap_values','plot_tft_shap','draw_shap_all_timesteps_plot','get_shap_values','plot_shap'
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

def get_tft_shap_values(run_id, X_test: pd.DataFrame, max_encoder_length=12):
    from src.trainers.tft_model import load_tft_checkpoint
    from src.trainers.tft_dataset import load_dataset_template, from_train_template
    logging.info("Loading TFT model...")
    model_path = os.path.join(RESULTS_PATH, run_id, "final", "best.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFT model checkpoint not found: {model_path}")

    model = load_tft_checkpoint(run_id)
    model.eval()

    from src.utils.utils import load_session_state
    session_state = load_session_state(run_id)
    features = session_state.get("features")
    targets = session_state.get("targets")

    # Load dataset template (create if missing for older runs)
    logging.info("Loading TFT dataset template...")
    try:
        train_template = load_dataset_template(run_id)
    except FileNotFoundError:
        logging.warning("Dataset template not found. Recreating with model encoders...")
        train_template = _create_template_with_model_encoders(model, session_state, run_id)

    # Filter out sequences that are too short for TFT requirements
    from configs.models.tft import TFTDatasetConfig
    config = TFTDatasetConfig()
    min_required_length = config.max_encoder_length + config.max_prediction_length

    # Group by series and filter by sequence length
    group_cols = ['Model', 'Scenario', 'Region']
    group_sizes = X_test.groupby(group_cols).size()
    sufficient_groups = group_sizes[group_sizes >= min_required_length].index

    if len(sufficient_groups) == 0:
        raise ValueError(f"No sequences meet minimum length requirement ({min_required_length} timesteps). "
                        f"Available sequence lengths: {group_sizes.min()}-{group_sizes.max()}")

    # Filter data to only include sufficient sequences
    mask = X_test.set_index(group_cols).index.isin(sufficient_groups)
    X_test_filtered = X_test[mask].reset_index(drop=True)

    logging.info(f"Filtered data: {len(sufficient_groups)}/{len(group_sizes)} sequences "
                f"meet minimum length requirement ({min_required_length} timesteps)")

    # Create test dataset from template
    test_dataset = from_train_template(train_template, X_test_filtered, mode="predict")

    # Create background data (smaller subset for SHAP)
    background_size = min(200, len(X_test_filtered))
    background_data = X_test_filtered.iloc[:background_size]
    background_dataset = from_train_template(train_template, background_data, mode="predict")

    # Get data loaders
    background_loader = background_dataset.to_dataloader(
        train=False, batch_size=50, num_workers=1, persistent_workers=False
    )
    test_loader = test_dataset.to_dataloader(
        train=False, batch_size=100, num_workers=1, persistent_workers=False
    )

    # Extract inputs for SHAP
    def extract_inputs_from_loader(loader, max_batches=1):
        inputs = []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            # TFT batch structure: x is dict with 'encoder_cont', 'encoder_cat', etc.
            x, _ = batch
            if isinstance(x, dict):
                # Combine encoder continuous and categorical features
                encoder_cont = x.get('encoder_cont', torch.empty(0))  # Shape: (batch, time, features)
                encoder_cat = x.get('encoder_cat', torch.empty(0))    # Shape: (batch, time, cat_features)

                if encoder_cont.numel() > 0 and encoder_cat.numel() > 0:
                    # Convert categorical to float and concatenate
                    encoder_cat_float = encoder_cat.float()
                    combined = torch.cat([encoder_cont, encoder_cat_float], dim=-1)
                elif encoder_cont.numel() > 0:
                    combined = encoder_cont
                elif encoder_cat.numel() > 0:
                    combined = encoder_cat.float()
                else:
                    raise ValueError("No encoder inputs found in batch")

                inputs.append(combined)

        if not inputs:
            raise ValueError("No inputs extracted from data loader")
        return torch.cat(inputs, dim=0)

    background_inputs = extract_inputs_from_loader(background_loader, max_batches=1)
    test_inputs = extract_inputs_from_loader(test_loader, max_batches=1)

    # Calculate SHAP values for each target separately
    all_temporal_shap = []
    all_averaged_shap = []

    for target_idx, target_name in enumerate(targets):
        logging.info(f"Calculating SHAP values for target {target_idx + 1}/{len(targets)}: {target_name}")

        # Create wrapper for this specific target
        import torch as _torch
        class TFTWrapperForSHAP(_torch.nn.Module):
            def __init__(self, tft_model, train_template, target_idx):
                super().__init__()
                self.tft_model = tft_model
                self.train_template = train_template
                self.target_idx = target_idx

            def forward(self, x):
                # x is already the combined encoder features (cont + cat) from the TFT dataset
                # We need to recreate the batch dict that TFT expects
                batch_size, time_steps, n_features = x.shape

                # Get the original batch structure from a sample batch
                sample_loader = test_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
                sample_batch = next(iter(sample_loader))
                sample_x, _ = sample_batch

                # Use the sample batch structure but replace with our SHAP input
                batch_dict = {}
                for key, value in sample_x.items():
                    if key == 'encoder_cont':
                        # Use the continuous part of our input
                        n_cont = value.shape[-1]
                        if n_cont > 0:
                            batch_dict[key] = x[:, :, :n_cont]
                        else:
                            batch_dict[key] = _torch.empty(batch_size, time_steps, 0)
                    elif key == 'encoder_cat':
                        # Use the categorical part of our input
                        n_cont = sample_x['encoder_cont'].shape[-1] if 'encoder_cont' in sample_x else 0
                        n_cat = value.shape[-1] if value.numel() > 0 else 0
                        if n_cat > 0 and n_features > n_cont:
                            batch_dict[key] = x[:, :, n_cont:n_cont+n_cat].long()
                        else:
                            batch_dict[key] = _torch.empty(batch_size, time_steps, 0, dtype=_torch.long)
                    else:
                        # Keep other keys as-is but adjust batch size
                        if isinstance(value, list):
                            batch_dict[key] = value
                        elif hasattr(value, 'size') and value.size(0) == 1:
                            batch_dict[key] = value.expand(batch_size, *value.shape[1:])
                        elif hasattr(value, 'size'):
                            batch_dict[key] = value[:batch_size]
                        else:
                            batch_dict[key] = value

                output = self.tft_model(batch_dict)

                # Extract prediction tensor
                pred_tensor = None
                if isinstance(output, _torch.Tensor):
                    pred_tensor = output
                elif isinstance(output, (list, tuple)):
                    for item in output:
                        if isinstance(item, _torch.Tensor):
                            pred_tensor = item
                            break
                elif hasattr(output, 'prediction'):
                    pred_tensor = output.prediction
                elif hasattr(output, 'output'):
                    pred_tensor = output.output
                else:
                    # Try to extract tensor attributes
                    for attr_name in dir(output):
                        if not attr_name.startswith('_'):
                            try:
                                attr_val = getattr(output, attr_name)
                                if isinstance(attr_val, _torch.Tensor):
                                    pred_tensor = attr_val
                                    break
                            except:
                                continue

                if pred_tensor is None:
                    raise ValueError(f"Could not extract tensor from TFT output of type {type(output)}")

                # For target-specific SHAP: need to extract predictions for specific target
                # TFT output shape: [batch, time_steps, attention_heads, lstm_layers]
                # We need to figure out how to map this to actual target predictions

                # For now, average over time steps and attention heads, take one lstm layer
                # This is a simplified approach - ideally we'd understand the exact TFT output format
                if pred_tensor.dim() == 4:
                    # Average over time steps (dim 1) and attention heads (dim 2)
                    # Use self.target_idx to select relevant component
                    result = pred_tensor.mean(dim=(1, 2))  # [batch, lstm_layers]
                    # Use target_idx to select which lstm layer or component
                    if result.shape[1] > self.target_idx:
                        return result[:, self.target_idx:self.target_idx+1]  # [batch, 1]
                    else:
                        # If not enough components, use sum of all
                        return result.sum(dim=1, keepdim=True)  # [batch, 1]
                else:
                    # For other shapes, just average to get [batch, 1]
                    while pred_tensor.dim() > 2:
                        pred_tensor = pred_tensor.mean(dim=-1)
                    if pred_tensor.dim() == 2:
                        return pred_tensor.mean(dim=1, keepdim=True)  # [batch, 1]
                    else:
                        return pred_tensor.unsqueeze(1)  # [batch, 1]

        wrapper = TFTWrapperForSHAP(model, train_template, target_idx)
        wrapper.eval()

        background_inputs.requires_grad_(True)
        test_inputs.requires_grad_(True)

        explainer = shap.DeepExplainer(wrapper, background_inputs)
        logging.info(f"Calculating SHAP values for {target_name}...")
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

        all_temporal_shap.append(shap_values)
        all_averaged_shap.append(_np.mean(shap_values, axis=1))

    # Combine all targets into final arrays
    original_temporal_shap = _np.concatenate(all_temporal_shap, axis=-1)  # [samples, time, features, targets]
    averaged = _np.concatenate(all_averaged_shap, axis=-1)  # [samples, features, targets]

    os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots"), exist_ok=True)
    _np.save(os.path.join(RESULTS_PATH, run_id, "plots", "tft_shap_values_temporal.npy"), original_temporal_shap)
    _np.save(os.path.join(RESULTS_PATH, run_id, "plots", "tft_shap_values.npy"), averaged)

    n_samples = averaged.shape[0]
    X_processed = X_test_filtered.iloc[:min(len(test_inputs), n_samples)][features]
    test_inputs_np = _to_numpy(test_inputs)

    return original_temporal_shap, averaged, X_processed, test_inputs_np

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
        draw_shap_all_timesteps_plot(run_id, temporal_shap, test_seq, features, targets, sequence_length, feature_name_map, model_type="lstm")
        draw_temporal_shap_plot(run_id, temporal_shap, X_proc, features, targets, sequence_length, feature_name_map, model_type="lstm")
    except Exception as e:
        logging.error("Failed to create LSTM SHAP plots: %s", e)
        logging.exception("Full error traceback:")

def plot_tft_shap(run_id, X_test_with_index: pd.DataFrame, features: List[str], targets: List[str], max_encoder_length=12, feature_name_map: Optional[Dict[str, str]] = None):
    logging.info("Creating TFT SHAP plots...")
    model_path = os.path.join(RESULTS_PATH, run_id, "final", "best.ckpt")
    if not os.path.exists(model_path):
        logging.warning("Skipping TFT SHAP plots: model checkpoint not found at %s", model_path)
        return

    # For TFT, we need to keep grouping columns for sequence filtering
    # Keep all columns that TFT needs: features, targets, group_ids, categorical columns, time_idx
    from configs.data import CATEGORICAL_COLUMNS, INDEX_COLUMNS
    from configs.models.tft import TFTDatasetConfig

    config = TFTDatasetConfig()
    # Include all columns that TFT needs: features, targets, group_ids, categoricals, time_idx, and time-varying columns
    time_known = ["Year", "DeltaYears"]  # From TFT config
    required_columns = set(features + targets + config.group_ids + CATEGORICAL_COLUMNS + [config.time_idx] + time_known)
    available_columns = [col for col in required_columns if col in X_test_with_index.columns]
    X_test = X_test_with_index[available_columns].reset_index(drop=True)

    import numpy as _np
    # Sample complete sequences instead of random rows to preserve temporal structure
    group_cols = ['Model', 'Scenario', 'Region']
    unique_groups = X_test.groupby(group_cols).size()

    # Target around 50-100 complete sequences for SHAP analysis
    max_sequences = min(100, len(unique_groups))
    if len(unique_groups) > max_sequences:
        # Randomly sample complete sequences
        sampled_groups = _np.random.choice(len(unique_groups), max_sequences, replace=False)
        selected_group_keys = unique_groups.iloc[sampled_groups].index

        # Filter data to include only selected sequences
        mask = X_test.set_index(group_cols).index.isin(selected_group_keys)
        X_test = X_test[mask].reset_index(drop=True)

        logging.info(f"Sampled {max_sequences} complete sequences from {len(unique_groups)} available sequences")

    try:
        temporal_shap, averaged, X_proc, test_seq = get_tft_shap_values(run_id, X_test, max_encoder_length)
        sequence_length = test_seq.shape[1]  # Get actual sequence length from TFT data
        draw_shap_all_timesteps_plot(run_id, temporal_shap, test_seq, features, targets, sequence_length, feature_name_map, model_type="tft")
        draw_temporal_shap_plot(run_id, temporal_shap, X_proc, features, targets, sequence_length, feature_name_map, model_type="tft")
    except Exception as e:
        logging.error("Failed to create TFT SHAP plots: %s", e)
        logging.exception("Full error traceback:")

def draw_shap_all_timesteps_plot(run_id: str, temporal_shap_values, test_sequences_np, features: List[str], targets: List[str], sequence_length: int, feature_name_map: Optional[Dict[str, str]] = None, model_type: str = "lstm") -> None:
    import numpy as _np
    if sequence_length <= 0:
        logging.warning("Invalid sequence_length=%d; skipping all-timesteps SHAP plot", sequence_length)
        return
    x_flat_parts = [test_sequences_np[:, t, :] for t in range(sequence_length)]
    X_flat = _np.concatenate(x_flat_parts, axis=1)
    feature_name_map = feature_name_map or {}
    base_names = [feature_name_map.get(f, f) for f in features]
    # Create timestep-prefixed feature names for proper temporal differentiation
    display_names = []
    for t in range(sequence_length):
        timestep_features = [f"timestep_{t}_{f}" for f in features]
        display_names.extend(build_feature_display_names(timestep_features, name_map=feature_name_map))
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
        ax.set_title(f"Impact on {targets[i]} ({OUTPUT_UNITS[i]})")
    fig.tight_layout()
    os.makedirs(os.path.join(RESULTS_PATH, run_id, 'plots'), exist_ok=True)
    fig.savefig(os.path.join(RESULTS_PATH, run_id, 'plots', f'{model_type}_shap_plot.png'))
    plt.close(fig)

def draw_temporal_shap_plot(run_id: str, temporal_shap_values, X_test: pd.DataFrame, features: List[str], targets: List[str], sequence_length: int, feature_name_map: Optional[Dict[str, str]] = None, model_type: str = "lstm") -> None:
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
    labels = build_feature_display_names([f"timestep_{t}" for t in range(sequence_length)])
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
    fig.savefig(os.path.join(RESULTS_PATH, run_id, 'plots', f'{model_type}_temporal_shap_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    create_timestep_comparison_plots(run_id, temporal_shap_values, features, targets, sequence_length, feature_name_map, model_type)

def create_timestep_comparison_plots(run_id: str, temporal_shap_values, features: List[str], targets: List[str], sequence_length: int, feature_name_map: Optional[Dict[str, str]] = None, model_type: str = "lstm") -> None:
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
        labels = build_feature_display_names([f"timestep_{t}" for t in range(sequence_length)])
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
    fig.savefig(os.path.join(RESULTS_PATH, run_id, 'plots', f'{model_type}_timestep_importance.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info("Temporal SHAP plots saved")

# Generic wrapper functions for model-agnostic usage
def get_shap_values(run_id, X_test: pd.DataFrame, model_type: str = "auto", **kwargs):
    """Get SHAP values for any supported model type (auto-detects if not specified)."""
    if model_type == "auto":
        # Auto-detect model type based on checkpoint location
        lstm_path = os.path.join(RESULTS_PATH, run_id, "final", "best.ckpt")
        tft_path = os.path.join(RESULTS_PATH, run_id, "final", "best.ckpt")

        # Check for TFT-specific files first
        dataset_template_path = os.path.join(RESULTS_PATH, run_id, "final", "dataset_template.pt")
        if os.path.exists(dataset_template_path):
            model_type = "tft"
        elif os.path.exists(lstm_path):
            model_type = "lstm"
        else:
            raise FileNotFoundError(f"No supported model checkpoint found for run_id: {run_id}")

    if model_type == "lstm":
        sequence_length = kwargs.get("sequence_length", 1)
        return get_lstm_shap_values(run_id, X_test, sequence_length)
    elif model_type == "tft":
        max_encoder_length = kwargs.get("max_encoder_length", 12)
        return get_tft_shap_values(run_id, X_test, max_encoder_length)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def plot_shap(run_id, X_test_with_index: pd.DataFrame, features: List[str], targets: List[str], model_type: str = "auto", feature_name_map: Optional[Dict[str, str]] = None, **kwargs):
    """Plot SHAP values for any supported model type (auto-detects if not specified)."""
    if model_type == "auto":
        # Auto-detect model type
        dataset_template_path = os.path.join(RESULTS_PATH, run_id, "final", "dataset_template.pt")
        if os.path.exists(dataset_template_path):
            model_type = "tft"
        else:
            model_type = "lstm"

    if model_type == "lstm":
        sequence_length = kwargs.get("sequence_length", 1)
        plot_lstm_shap(run_id, X_test_with_index, features, targets, sequence_length, feature_name_map)
    elif model_type == "tft":
        max_encoder_length = kwargs.get("max_encoder_length", 12)
        plot_tft_shap(run_id, X_test_with_index, features, targets, max_encoder_length, feature_name_map)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Legacy function name for backward compatibility
def draw_lstm_all_timesteps_shap_plot(run_id: str, temporal_shap_values, test_sequences_np, features: List[str], targets: List[str], sequence_length: int, feature_name_map: Optional[Dict[str, str]] = None) -> None:
    """Legacy function - use draw_shap_all_timesteps_plot instead."""
    draw_shap_all_timesteps_plot(run_id, temporal_shap_values, test_sequences_np, features, targets, sequence_length, feature_name_map, model_type="lstm")

# Helper functions for TFT SHAP with older models
def _extract_categorical_encoders_from_model(model):
    """Extract categorical encoders from a trained TFT model."""
    encoders = {}
    try:
        # Try different ways to access encoders from the model
        if hasattr(model, 'hparams') and hasattr(model.hparams, 'categorical_encoders'):
            encoders = model.hparams.categorical_encoders
        elif hasattr(model, 'categorical_encoders'):
            encoders = model.categorical_encoders
        elif hasattr(model, 'dataset_parameters') and 'categorical_encoders' in model.dataset_parameters:
            encoders = model.dataset_parameters['categorical_encoders']

        logging.info(f"Successfully extracted {len(encoders)} categorical encoders from model")
        return encoders
    except Exception as e:
        logging.warning(f"Could not extract categorical encoders from model: {e}")
        return {}

def _create_template_with_model_encoders(model, session_state, run_id):
    """Create a dataset template using encoders extracted from the trained model."""
    from src.trainers.tft_dataset import create_train_dataset

    # Extract encoders from the model
    model_encoders = _extract_categorical_encoders_from_model(model)

    if model_encoders:
        logging.info("Using extracted encoders from model for dataset template")
        # Create a modified session state with the model's encoders
        session_state_copy = session_state.copy()
        # We'll need to modify the dataset creation to use these encoders
        return _create_dataset_with_custom_encoders(session_state_copy, model_encoders)
    else:
        logging.warning("No encoders found in model, using default dataset creation")
        # Fall back to original method
        train_template, _ = create_train_dataset(session_state)
        return train_template

def _create_dataset_with_custom_encoders(session_state, custom_encoders):
    """Create a TFT dataset using custom categorical encoders."""
    from src.trainers.tft_dataset import create_train_dataset
    from configs.models.tft import TFTDatasetConfig

    try:
        # Temporarily modify the TFT config to use our custom encoders
        original_build = TFTDatasetConfig.build

        def build_with_custom_encoders(self, features, targets, mode):
            params = original_build(self, features, targets, mode)
            params['categorical_encoders'] = custom_encoders
            return params

        # Monkey patch the build method
        TFTDatasetConfig.build = build_with_custom_encoders

        # Create the dataset template
        train_template, _ = create_train_dataset(session_state)

        # Restore the original build method
        TFTDatasetConfig.build = original_build

        logging.info("Successfully created dataset template with custom encoders")
        return train_template

    except Exception as e:
        logging.error(f"Failed to create dataset with custom encoders: {e}")
        # Restore original method just in case
        TFTDatasetConfig.build = original_build
        # Fall back to default creation
        train_template, _ = create_train_dataset(session_state)
        return train_template
