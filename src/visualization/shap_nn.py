# Neural network SHAP plotting (migrated from utils.plot_shap_nn)
import os, logging, numpy as np, pandas as pd, shap, torch
from typing import List, Optional, Dict, Iterable, Set
from configs.paths import RESULTS_PATH
from src.utils.utils import get_run_root
from configs.data import CATEGORICAL_COLUMNS, NON_FEATURE_COLUMNS, OUTPUT_UNITS
from configs.visualization import DEFAULT_REGION
from .helpers import (
    make_grid,
    render_external_plot,
    build_feature_display_names,
    draw_shap_beeswarm,
    filter_index_frame_by_region,
    sample_scenario_groups,
)

__all__ = [
    'get_lstm_shap_values','plot_lstm_shap','draw_lstm_all_timesteps_shap_plot','draw_temporal_shap_plot','create_timestep_comparison_plots',
    'get_tft_shap_values','plot_tft_shap','draw_shap_all_timesteps_plot','get_shap_values'
]


def _derive_tft_feature_names(
    train_template,
    session_features: Optional[List[str]],
    available_columns: Optional[Iterable[str]] = None,
    sample_batch: Optional[dict] = None,
) -> List[str]:
    """Build an encoder-aligned feature list from a saved TFT dataset template."""

    base_features = list(session_features or [])
    available_list = list(available_columns) if available_columns is not None else None

    def _feature_count(batch: Optional[dict], key: str) -> Optional[int]:
        if not isinstance(batch, dict):
            return None
        tensor = batch.get(key)
        if tensor is None:
            return None
        if hasattr(tensor, "numel") and tensor.numel() == 0:
            return 0
        if hasattr(tensor, "shape") and len(tensor.shape) >= 3:
            return int(tensor.shape[-1])
        return None

    cont_len = _feature_count(sample_batch, "encoder_cont")
    cat_len = _feature_count(sample_batch, "encoder_cat")

    def _unique(names: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        for name in names:
            if name is None:
                continue
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    cont_names: List[str] = []
    cat_names: List[str] = []

    if train_template is not None:
        try:
            cont_names = _unique(getattr(train_template, "reals", []))
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Failed to derive continuous feature names from TFT template: %s",
                exc,
            )
            cont_names = []
        try:
            cat_names = _unique(getattr(train_template, "categoricals", []))
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Failed to derive categorical feature names from TFT template: %s",
                exc,
            )
            cat_names = []

    fallback_cont = [f for f in base_features if f not in CATEGORICAL_COLUMNS]
    fallback_cat = [f for f in base_features if f in CATEGORICAL_COLUMNS]

    def _coerce(names: List[str], target_len: Optional[int], fallback: List[str], prefix: str) -> List[str]:
        if target_len is None:
            return names
        trimmed = list(names[: target_len])
        if len(trimmed) < target_len:
            for candidate in fallback:
                if candidate not in trimmed:
                    trimmed.append(candidate)
                if len(trimmed) == target_len:
                    break
        while len(trimmed) < target_len:
            trimmed.append(f"{prefix}_{len(trimmed)}")
        return trimmed

    cont_names = _coerce(cont_names, cont_len, fallback_cont, "encoder_cont")
    cat_names = _coerce(cat_names, cat_len, fallback_cat, "encoder_cat")

    derived = cont_names + cat_names
    if derived:
        return derived
    if available_list:
        return available_list
    return base_features


def _align_feature_names(feature_names: Optional[List[str]], feature_count: int) -> List[str]:
    """Ensure the feature name list matches the encoded feature dimension."""
    aligned = list(feature_names or [])
    if feature_count <= 0:
        return aligned
    if len(aligned) < feature_count:
        start = len(aligned)
        for idx in range(start, feature_count):
            aligned.append(f"feature_{idx}")
        logging.warning(
            "Padded feature names with %d placeholders to match TFT encoder dimension %d.",
            feature_count - start,
            feature_count,
        )
    elif len(aligned) > feature_count:
        logging.info(
            "Trimming TFT feature list from %d to %d to match encoder dimension.",
            len(aligned),
            feature_count,
        )
        aligned = aligned[:feature_count]
    return aligned

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
    model_path = os.path.join(get_run_root(run_id), "final", "best.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model checkpoint not found: {model_path}")
    model = LSTMModel.load_from_checkpoint(model_path)
    model.eval()
    from src.utils.run_store import RunStore
    store = RunStore(run_id)
    scaler_X = store.load_artifact("lstm_scaler_X.pkl")
    features, targets = store.load_features()
    categorical_features = []
    if store.has_train_meta():
        meta = store.load_train_meta()
        features = meta.get("lstm_raw_features", features)
        categorical_features = meta.get("lstm_categorical_features", [])
    continuous_features = [f for f in features if f not in categorical_features]

    def preprocess_features(data, continuous_features, categorical_features, scaler_X, mask_value=-1.0):
        from configs.data import CATEGORICAL_COLUMNS, REGION_CATEGORIES
        # Scale continuous features only
        X_cont = data[continuous_features].copy() if continuous_features else pd.DataFrame(index=data.index)
        X_cont_filled = X_cont.fillna(mask_value).astype(np.float32)
        X_cont_scaled = scaler_X.transform(X_cont_filled) if len(continuous_features) > 0 else X_cont_filled.values

        # Extract categorical codes — already int-encoded by derive_splits
        cat_codes = {}
        for col in categorical_features:
            if col not in data.columns:
                continue
            if pd.api.types.is_integer_dtype(data[col]):
                # Already encoded upstream (derive_splits)
                cat_codes[col] = data[col].values.astype(np.int64)
            elif col == 'Region':
                cat_codes[col] = (
                    pd.Categorical(data[col].astype(str), categories=REGION_CATEGORIES, ordered=True)
                    .codes.astype(np.int64)
                )
            else:
                cat_codes[col] = data[col].astype('category').cat.codes.values.astype(np.int64)
        X_cat = np.column_stack([cat_codes[c] for c in categorical_features]) if categorical_features else np.empty((len(data), 0), dtype=np.int64)
        return X_cont_scaled, X_cat

    def create_sequences(X_cont_scaled, X_cat, seq_len):
        import torch as _torch
        cont_seqs, cat_seqs = [], []
        for i in range(len(X_cont_scaled) - seq_len + 1):
            cont_seqs.append(_torch.FloatTensor(X_cont_scaled[i:i+seq_len]))
            cat_seqs.append(_torch.LongTensor(X_cat[i:i+seq_len]))
        if cont_seqs:
            return _torch.stack(cont_seqs), _torch.stack(cat_seqs)
        return _torch.empty(0, seq_len, X_cont_scaled.shape[1]), _torch.empty(0, seq_len, X_cat.shape[1], dtype=_torch.long)

    background_size = min(200, len(X_test))
    bg_cont, bg_cat = preprocess_features(X_test.iloc[:background_size], continuous_features, categorical_features, scaler_X)
    bg_cont_seq, bg_cat_seq = create_sequences(bg_cont, bg_cat, sequence_length)
    if len(bg_cont_seq) > 50:
        import numpy as _np
        idx = _np.random.choice(len(bg_cont_seq), 50, replace=False)
        background_data = bg_cont_seq[idx]
        background_cat = bg_cat_seq[idx]
    else:
        background_data = bg_cont_seq
        background_cat = bg_cat_seq
    test_size = min(100, len(X_test))
    test_cont, test_cat = preprocess_features(X_test.iloc[:test_size], continuous_features, categorical_features, scaler_X)
    test_inputs, test_cat_seq = create_sequences(test_cont, test_cat, sequence_length)
    import torch as _torch

    # SHAP explains continuous features only; categorical embeddings are held fixed
    class LSTMWrapperForSHAP(_torch.nn.Module):
        def __init__(self, lstm_model, seq_len, fixed_cat):
            super().__init__()
            self.lstm_model = lstm_model
            self.seq_len = seq_len
            self.fixed_cat = fixed_cat  # [batch, seq_len, num_cat] — expanded per sample
        def forward(self, x):
            batch_size = x.shape[0]
            mask = _torch.ones(batch_size, self.seq_len, dtype=_torch.float32, device=x.device)
            # Use the fixed categorical indices (broadcast if needed)
            cat = self.fixed_cat[:batch_size] if self.fixed_cat.shape[0] >= batch_size else self.fixed_cat.expand(batch_size, -1, -1)
            return self.lstm_model(x, mask=mask, teacher_forcing=False, cat_indices=cat)
    wrapper = LSTMWrapperForSHAP(model, sequence_length, background_cat)
    wrapper.eval()
    background_data.requires_grad_(True)
    test_inputs.requires_grad_(True)
    explainer = shap.DeepExplainer(wrapper, background_data)
    # Swap to test categorical indices for explanation pass
    wrapper.fixed_cat = test_cat_seq
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
    os.makedirs(os.path.join(get_run_root(run_id), "plots"), exist_ok=True)
    _np.save(os.path.join(get_run_root(run_id), "plots", "lstm_shap_values_temporal.npy"), original_temporal_shap)
    _np.save(os.path.join(get_run_root(run_id), "plots", "lstm_shap_values.npy"), averaged)
    n_samples = averaged.shape[0]
    X_processed = X_test.iloc[:min(test_size, n_samples)].loc[:, continuous_features]
    test_sequences_np = _to_numpy(test_inputs)
    return original_temporal_shap, averaged, X_processed, test_sequences_np

def get_tft_shap_values(
    run_id,
    X_test: pd.DataFrame,
    max_encoder_length=12,
    *,
    use_cached: bool = True,
):
    from src.trainers.tft_model import load_tft_checkpoint
    from src.trainers.tft_dataset import load_dataset_template, from_train_template
    logging.info("Loading TFT model...")
    model_path = os.path.join(get_run_root(run_id), "final", "best.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFT model checkpoint not found: {model_path}")

    model = load_tft_checkpoint(run_id)
    model.eval()

    from src.utils.run_store import RunStore
    store = RunStore(run_id)
    session_features, targets = store.load_features()
    session_features = list(session_features)

    plots_dir = os.path.join(get_run_root(run_id), "plots")
    cache_path = os.path.join(plots_dir, "tft_shap_cache.npz")

    if use_cached and os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=True) as cached:
                original_temporal_shap = cached["temporal_shap"]
                averaged = cached["averaged_shap"]
                test_inputs_np = cached["test_inputs"]
                base_matrix = cached["base_matrix"]
                feature_names = cached["feature_names"].tolist()
                feature_columns = cached["feature_columns"].tolist()
            X_processed = pd.DataFrame(base_matrix, columns=feature_columns)
            logging.info("Loaded cached TFT SHAP data from %s", cache_path)
            return original_temporal_shap, averaged, X_processed, test_inputs_np, feature_names
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Failed to load cached TFT SHAP data from %s: %s", cache_path, exc
            )

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
    test_dataset = from_train_template(train_template, X_test_filtered, mode="eval")

    # Create background data (smaller subset for SHAP)
    background_size = min(200, len(X_test_filtered))
    background_data = X_test_filtered.iloc[:background_size]
    background_dataset = from_train_template(train_template, background_data, mode="eval")

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

    # Extract sample batch structure once for efficiency (avoids recreating DataLoader in forward pass)
    sample_loader = test_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    sample_batch = next(iter(sample_loader))
    sample_x, _ = sample_batch

    # Derive feature names from dataset and encoder structure
    features = _derive_tft_feature_names(
        train_template,
        session_features,
        available_columns=X_test_filtered.columns,
        sample_batch=sample_x,
    )

    # Calculate SHAP values for each target separately
    all_temporal_shap = []
    all_averaged_shap = []

    for target_idx, target_name in enumerate(targets):
        logging.info(f"Calculating SHAP values for target {target_idx + 1}/{len(targets)}: {target_name}")

        # Create wrapper for this specific target
        import torch as _torch
        class TFTWrapperForSHAP(_torch.nn.Module):
            def __init__(self, tft_model, train_template, target_idx, sample_batch_structure):
                super().__init__()
                self.tft_model = tft_model
                self.train_template = train_template
                self.target_idx = target_idx
                self.sample_batch_structure = sample_batch_structure

            def forward(self, x):
                # x is already the combined encoder features (cont + cat) from the TFT dataset
                # We need to recreate the batch dict that TFT expects
                batch_size, time_steps, n_features = x.shape

                # Use the pre-extracted sample batch structure
                sample_x = self.sample_batch_structure

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

                # Move all tensors to the model's device
                device = next(self.tft_model.parameters()).device
                batch_dict = {
                    k: v.to(device) if isinstance(v, _torch.Tensor) else v
                    for k, v in batch_dict.items()
                }

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

        wrapper = TFTWrapperForSHAP(model, train_template, target_idx, sample_x)
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

    os.makedirs(plots_dir, exist_ok=True)
    _np.save(os.path.join(plots_dir, "tft_shap_values_temporal.npy"), original_temporal_shap)
    _np.save(os.path.join(plots_dir, "tft_shap_values.npy"), averaged)

    test_inputs_np = _to_numpy(test_inputs)
    feature_count = test_inputs_np.shape[-1] if test_inputs_np.ndim >= 2 else 0
    features = _align_feature_names(features, feature_count)

    if test_inputs_np.ndim == 3:
        base_matrix = test_inputs_np[:, 0, :feature_count]
    elif test_inputs_np.ndim == 2:
        base_matrix = test_inputs_np[:, :feature_count]
    else:
        base_matrix = test_inputs_np.reshape(test_inputs_np.shape[0], -1) if test_inputs_np.size else _np.empty((0, feature_count))

    X_processed = pd.DataFrame(base_matrix, columns=features)

    try:
        cache_payload = {
            "temporal_shap": original_temporal_shap,
            "averaged_shap": averaged,
            "test_inputs": test_inputs_np,
            "base_matrix": base_matrix,
            "feature_names": np.array(features, dtype=object),
            "feature_columns": np.array(features, dtype=object),
        }
        np.savez_compressed(cache_path, **cache_payload)
        logging.info("Saved TFT SHAP cache to %s", cache_path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to write TFT SHAP cache to %s: %s", cache_path, exc)

    return original_temporal_shap, averaged, X_processed, test_inputs_np, features

def plot_lstm_shap(run_id, X_test_with_index: pd.DataFrame, features: List[str], targets: List[str], sequence_length=1, region: Optional[str] = DEFAULT_REGION):
    logging.info("Creating LSTM SHAP plots...")
    model_path = os.path.join(get_run_root(run_id), "final", "best.ckpt")
    if not os.path.exists(model_path):
        logging.warning("Skipping LSTM SHAP plots: model checkpoint not found at %s", model_path)
        return
    # Optional region filter before scenario grouping (works on index frame)
    X_filtered, _, pre_rows, post_rows, matched, _mode = filter_index_frame_by_region(
        X_test_with_index,
        region,
        log_prefix="Applied region filter",
    )

    # Scenario-based sampling on the full index frame (Model/Scenario kept as indices)
    group_keys, total_groups, used_groups, group_cols = sample_scenario_groups(
        X_filtered,
        log_prefix="LSTM SHAP",
    )

    if group_cols and not group_keys.empty:
        # Inner-join to keep all rows belonging to sampled groups
        X_joined = X_filtered.merge(group_keys, on=group_cols, how="inner")
    else:
        X_joined = X_filtered

    X_test = X_joined.drop(columns=NON_FEATURE_COLUMNS, errors="ignore").reset_index(drop=True)

    logging.info(
        "LSTM SHAP: %d rows after region+scenario filtering (%d -> %d groups by %s)",
        X_test.shape[0],
        total_groups,
        used_groups,
        ",".join(group_cols) if group_cols else "<none>",
    )
    try:
        temporal_shap, averaged, X_proc, test_seq = get_lstm_shap_values(run_id, X_test, sequence_length)
        draw_shap_all_timesteps_plot(run_id, temporal_shap, test_seq, features, targets, sequence_length, model_type="lstm")
        draw_shap_all_timesteps_plot(run_id, temporal_shap, test_seq, features, targets, sequence_length, model_type="lstm", xlim_range=(-0.3, 0.5))
        draw_temporal_shap_plot(run_id, temporal_shap, pd.DataFrame(X_proc), features, targets, sequence_length, model_type="lstm")
    except Exception as e:
        logging.error("Failed to create LSTM SHAP plots: %s", e)
        logging.exception("Full error traceback:")

def plot_tft_shap(
    run_id,
    X_test_with_index: pd.DataFrame,
    features: List[str],
    targets: List[str],
    max_encoder_length=12,
    region: Optional[str] = DEFAULT_REGION,
    use_cached: bool = True,
):
    logging.info("Creating TFT SHAP plots...")
    model_path = os.path.join(get_run_root(run_id), "final", "best.ckpt")
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
    # Optional region filter prior to scenario sampling (robust, on index frame)
    X_filtered, _, pre_rows, post_rows, matched, _mode = filter_index_frame_by_region(
        X_test_with_index,
        region,
        log_prefix="Applied region filter",
    )

    # Scenario-based sampling on the filtered index frame using grouping columns
    group_keys, total_groups, used_groups, group_cols = sample_scenario_groups(
        X_filtered,
        log_prefix="TFT SHAP",
    )

    if group_cols and not group_keys.empty:
        X_joined = X_filtered.merge(group_keys, on=group_cols, how="inner")
    else:
        X_joined = X_filtered

    # Now restrict to TFT-required columns and selected rows only
    available_columns = [col for col in required_columns if col in X_joined.columns]
    X_test = X_joined[available_columns].reset_index(drop=True)

    if use_cached:
        cache_path = os.path.join(get_run_root(run_id), "plots", "tft_shap_cache.npz")
        if os.path.exists(cache_path):
            logging.info("Found cached TFT SHAP data at %s; will reuse it for plotting.", cache_path)

    try:
        temporal_shap, averaged, X_proc, test_seq, feature_names = get_tft_shap_values(
            run_id,
            X_test,
            max_encoder_length,
            use_cached=use_cached,
        )
        sequence_length = test_seq.shape[1]  # Get actual sequence length from TFT data
        draw_shap_all_timesteps_plot(run_id, temporal_shap, test_seq, feature_names, targets, sequence_length, model_type="tft")
        draw_shap_all_timesteps_plot(run_id, temporal_shap, test_seq, feature_names, targets, sequence_length, model_type="tft", xlim_range=(-0.3, 0.5))
        draw_temporal_shap_plot(run_id, temporal_shap, X_proc, feature_names, targets, sequence_length, model_type="tft")
    except Exception as e:
        logging.error("Failed to create TFT SHAP plots: %s", e)
        logging.exception("Full error traceback:")

def draw_shap_all_timesteps_plot(run_id: str, temporal_shap_values, test_sequences_np, features: List[str], targets: List[str], sequence_length: int, model_type: str = "lstm", xlim_range: Optional[tuple] = None) -> None:
    import numpy as _np
    if sequence_length <= 0:
        logging.warning("Invalid sequence_length=%d; skipping all-timesteps SHAP plot", sequence_length)
        return
    feature_count = temporal_shap_values.shape[2] if temporal_shap_values.ndim >= 3 else 0
    features = _align_feature_names(features, feature_count)

    # Aggregate SHAP values across timesteps: sum absolute contributions per feature
    # temporal_shap_values shape: [samples, timesteps, features, targets]
    # test_sequences_np shape: [samples, timesteps, features]
    # After aggregation: [samples, features] — one row per feature in beeswarm
    shap_agg = _np.sum(temporal_shap_values, axis=1)  # [samples, features, targets]
    X_agg = _np.mean(test_sequences_np, axis=1)        # [samples, features] — mean feature value for color

    display_names = build_feature_display_names(features)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    num_targets = len(targets)
    fig, axes = make_grid(num_targets, base_figsize=(20, 20))

    # Create directory for individual plots
    indiv_plots_dir = os.path.join(get_run_root(run_id), 'plots', 'indiv_plots', 'shap')
    os.makedirs(indiv_plots_dir, exist_ok=True)

    for i, ax in enumerate(axes):
        if i >= num_targets:
            ax.axis('off')
            continue
        def _plot(fig_local, _i=i):
            ax_local = fig_local.add_subplot(111)
            draw_shap_beeswarm(
                ax_local,
                shap_agg[:, :, _i],
                X_agg,
                display_names,
                max_display=8,
                xlim_range=xlim_range,
            )
            fig_local.tight_layout()
        render_external_plot(ax, _plot)
        ax.set_title(f"Impact on {targets[i]} ({OUTPUT_UNITS[i]})")

        # Save individual plot for this target
        fig_indiv = plt.figure(figsize=(10, 8))
        ax_indiv = fig_indiv.add_subplot(111)
        draw_shap_beeswarm(
            ax_indiv,
            shap_agg[:, :, i],
            X_agg,
            display_names,
            max_display=8,
            xlim_range=xlim_range,
        )
        plt.title(f"Impact on {targets[i]} ({OUTPUT_UNITS[i]})")
        plt.tight_layout()
        indiv_filename = f'{targets[i]}_match_xgb_range.png' if xlim_range is not None else f'{targets[i]}.png'
        fig_indiv.savefig(os.path.join(indiv_plots_dir, indiv_filename), dpi=300, bbox_inches='tight')
        plt.close(fig_indiv)

    fig.tight_layout()
    os.makedirs(os.path.join(get_run_root(run_id), 'plots'), exist_ok=True)
    filename = f'{model_type}_shap_plot_match_xgb_range.png' if xlim_range is not None else f'{model_type}_shap_plot.png'
    fig.savefig(os.path.join(get_run_root(run_id), 'plots', filename))
    plt.close(fig)

def draw_temporal_shap_plot(run_id: str, temporal_shap_values, X_test: pd.DataFrame, features: List[str], targets: List[str], sequence_length: int, model_type: str = "lstm") -> None:
    import matplotlib.pyplot as plt, numpy as _np
    from tqdm import tqdm
    plt.rcParams.update({'font.size': 12})
    num_targets = len(targets)
    fig, axes = make_grid(num_targets, base_figsize=(20, 20))
    feature_count = temporal_shap_values.shape[2] if temporal_shap_values.ndim >= 3 else 0
    features = _align_feature_names(features, feature_count)
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
    display_names = build_feature_display_names([features[idx] for idx in top_idx])
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([n[:30] + '...' if len(n) > 30 else n for n in display_names], fontsize=10)
    ax.set_title(f"Temporal SHAP: {targets[i]} ({OUTPUT_UNITS[i]})", fontsize=14)
    ax.set_xlabel("Time Step in Sequence", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    os.makedirs(os.path.join(get_run_root(run_id), 'plots'), exist_ok=True)
    fig.savefig(os.path.join(get_run_root(run_id), 'plots', f'{model_type}_temporal_shap_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    create_timestep_comparison_plots(run_id, temporal_shap_values, features, targets, sequence_length, model_type)

def create_timestep_comparison_plots(run_id: str, temporal_shap_values, features: List[str], targets: List[str], sequence_length: int, model_type: str = "lstm") -> None:
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
    fig.savefig(os.path.join(get_run_root(run_id), 'plots', f'{model_type}_timestep_importance.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info("Temporal SHAP plots saved")

# Generic wrapper functions for model-agnostic usage
def get_shap_values(run_id, X_test: pd.DataFrame, model_type: str = "auto", **kwargs):
    """Get SHAP values for any supported model type (auto-detects if not specified)."""
    if model_type == "auto":
        # Auto-detect model type based on checkpoint location
        lstm_path = os.path.join(get_run_root(run_id), "final", "best.ckpt")
        tft_path = os.path.join(get_run_root(run_id), "final", "dataset_template.pt")

        # Check for TFT-specific files first
        if os.path.exists(tft_path):
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
        use_cached = kwargs.get("use_cached", True)
        results = get_tft_shap_values(
            run_id,
            X_test,
            max_encoder_length,
            use_cached=use_cached,
        )
        return results
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def plot_nn_shap(run_id, X_test_with_index: pd.DataFrame, features: List[str], targets: List[str], model_type: str = "auto", **kwargs):
    """Plot SHAP values for any supported model type (auto-detects if not specified)."""
    if model_type == "lstm":
        sequence_length = kwargs.get("sequence_length", 1)
        region = kwargs.get("region", DEFAULT_REGION)
        # Apply region filter if provided (robust)
        X_input, _, _, _, _, _mode = filter_index_frame_by_region(
            X_test_with_index, region, log_prefix="Applied region filter"
        )
        temporal_shap, averaged, X_proc, test_seq = get_lstm_shap_values(run_id, X_input, sequence_length)
        sequence_length = test_seq.shape[1]
        draw_lstm_all_timesteps_shap_plot(run_id, temporal_shap, test_seq, features, targets, sequence_length)
        draw_temporal_shap_plot(run_id, temporal_shap, pd.DataFrame(X_proc), features, targets, sequence_length)
    elif model_type == "tft":
        max_encoder_length = kwargs.get("max_encoder_length", 12)
        region = kwargs.get("region", DEFAULT_REGION)
        use_cached = kwargs.get("use_cached", True)
        plot_tft_shap(
            run_id,
            X_test_with_index,
            features,
            targets,
            max_encoder_length,
            region=region,
            use_cached=use_cached,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Legacy function name for backward compatibility
def draw_lstm_all_timesteps_shap_plot(run_id: str, temporal_shap_values, test_sequences_np, features: List[str], targets: List[str], sequence_length: int, xlim_range: Optional[tuple] = None) -> None:
    """Legacy function - use draw_shap_all_timesteps_plot instead."""
    draw_shap_all_timesteps_plot(run_id, temporal_shap_values, test_sequences_np, features, targets, sequence_length, model_type="lstm", xlim_range=xlim_range)

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
    from src.trainers.tft_dataset import create_train_dataset, create_dataset_with_custom_encoders

    # Extract encoders from the model
    model_encoders = _extract_categorical_encoders_from_model(model)

    if model_encoders:
        logging.info("Using extracted encoders from model for dataset template")
        # Use the new factory function - no monkey patching needed!
        return create_dataset_with_custom_encoders(session_state, model_encoders)
    else:
        logging.warning("No encoders found in model, using default dataset creation")
        # Fall back to original method
        train_template, _ = create_train_dataset(session_state)
        return train_template

