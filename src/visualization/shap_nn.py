"""SHAP plots for the sequence models (LSTM and TFT).

Both models are explained with expected gradients -- the estimator behind
shap.GradientExplainer, see ``_expected_gradients`` -- over a seeded random
sample of windows, against a background drawn the same way.  Only continuous inputs are attributed: each explained window keeps its
own categorical codes -- and, for the TFT, its own lengths and target scale --
which are held fixed while the continuous inputs are interpolated
towards the background.
"""
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch

from src.utils.utils import get_run_root
from configs.data import CATEGORICAL_COLUMNS, INDEX_COLUMNS, NON_FEATURE_COLUMNS, SPLIT_SEED
from configs.visualization import (
    DEFAULT_REGION,
    SHAP_FONT_SIZE,
    SHAP_GRADIENT_BATCH_SIZE,
    SHAP_GRADIENT_NSAMPLES,
    SHAP_GRID_FIGSIZE,
    SHAP_INDIVIDUAL_FIGSIZE,
    SHAP_MAX_DISPLAY,
    SHAP_NN_BACKGROUND_WINDOWS,
    SHAP_NN_EXPLAINED_WINDOWS,
)
from .helpers import (
    make_grid,
    output_unit,
    render_external_plot,
    build_feature_display_names,
    draw_shap_beeswarm,
    filter_index_frame_by_region,
    sample_scenario_groups,
)

__all__ = [
    'get_lstm_shap_values', 'plot_lstm_shap', 'draw_temporal_shap_plot', 'create_timestep_comparison_plots',
    'get_tft_shap_values', 'plot_tft_shap', 'draw_shap_all_timesteps_plot',
    'draw_window_sample', 'normalise_temporal_shap', 'additivity_report', 'SHAP_CACHE_FORMAT',
]

# Layout of <model>_shap_cache.npz.  A cache written before the sampling and
# the TFT wrapper were fixed carries no format and is not read back.
SHAP_CACHE_FORMAT = 2


def _plots_dir(run_id: str, out_dir: Optional[str] = None) -> str:
    """Where SHAP arrays and plots go: *out_dir*, or the run's own plots directory."""
    return out_dir if out_dir else os.path.join(get_run_root(run_id), "plots")


def draw_window_sample(
    n_windows: int,
    n_explained: int = SHAP_NN_EXPLAINED_WINDOWS,
    n_background: int = SHAP_NN_BACKGROUND_WINDOWS,
    seed: int = SPLIT_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Seeded positions of the windows to explain and of the background.

    Both are drawn without replacement from all *n_windows*, so neither
    depends on the order the windows are stored in.  They are disjoint while
    there are enough windows for both; otherwise the background is drawn
    from all of them again.
    """
    if n_windows <= 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_windows)
    explained = np.sort(order[:n_explained])
    rest = order[n_explained:]
    if len(rest) >= n_background:
        background = np.sort(rest[:n_background])
    else:
        background = np.sort(rng.choice(n_windows, min(n_background, n_windows), replace=False))
    return explained, background


def normalise_temporal_shap(temporal_shap: np.ndarray) -> np.ndarray:
    """Put [samples, steps, features, targets] SHAP values on the shared scale.

    Each target is divided by its sum over features of mean |SHAP|, with SHAP
    summed over the steps first -- the per-feature quantity the beeswarm
    shows, and the one ``helpers.normalise_shap_values`` scales XGBoost's by.
    """
    temporal_shap = np.array(temporal_shap, dtype=float)
    totals = np.abs(temporal_shap.sum(axis=1)).mean(axis=0).sum(axis=0)  # [targets]
    return temporal_shap / np.where(totals > 0, totals, 1.0)


def additivity_report(temporal_shap: np.ndarray, fx: np.ndarray, base_values: np.ndarray, targets: Sequence[str]) -> Dict:
    """How well the attributions add up to the prediction they decompose.

    For every explained window and target, the SHAP values summed over steps
    and features should equal f(x) minus the mean of f over the background --
    with the window's own fixed context, which is what *base_values* holds.
    Reports the correlation between the two, pooled and per target, and the
    error relative to the spread of the differences.
    """
    total = np.asarray(temporal_shap).sum(axis=(1, 2))               # [samples, targets]
    delta = np.asarray(fx) - np.asarray(base_values)                 # [samples, targets]

    def corr(a, b) -> float:
        a, b = np.ravel(a), np.ravel(b)
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def relative_error(a, b) -> float:
        spread = float(np.sqrt(np.mean(np.square(b))))
        return float(np.sqrt(np.mean(np.square(a - b))) / spread) if spread > 0 else float("nan")

    # Targets differ in magnitude, so the pooled figure is computed on each
    # target's values divided by the spread of its own differences.
    scale = np.sqrt(np.mean(np.square(delta), axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    return {
        "n_explained": int(total.shape[0]),
        "correlation": corr(total / scale, delta / scale),
        "relative_error": relative_error(total / scale, delta / scale),
        "per_target": {
            str(t): {"correlation": corr(total[:, i], delta[:, i]), "relative_error": relative_error(total[:, i], delta[:, i])}
            for i, t in enumerate(targets)
        },
    }


def _log_additivity(model_name: str, report: Dict) -> None:
    worst = min(report["per_target"].items(), key=lambda kv: kv[1]["correlation"], default=(None, {"correlation": float("nan")}))
    logging.info(
        "%s SHAP additivity over %d windows: corr(sum SHAP, f(x) - background mean) = %.4f pooled "
        "(relative error %.3f); lowest target %s at %.4f",
        model_name, report["n_explained"], report["correlation"], report["relative_error"],
        worst[0], worst[1]["correlation"],
    )


def _expected_gradients(wrapper, row: torch.Tensor, background: torch.Tensor, n_samples: int, seed: int = SPLIT_SEED) -> np.ndarray:
    """Expected-gradients SHAP values of one window, shaped [steps, features, targets].

    The estimator shap.GradientExplainer implements: the mean over background
    windows b and positions a in [0, 1] of (x - b) * grad f(b + a (x - b)),
    whose expectation sums to f(x) minus the background mean of f.  Written
    out here because the library draws its samples one at a time in Python
    and runs a forward pass per output: on the TFT that is about 18 s per
    window at 500 samples.  Here a batch is one forward pass and one backward
    pass per output, and the draws are balanced rather than independent:
    *n_samples* is rounded up to a whole number of passes over the
    background, so every background window is used equally often, and each
    one's positions are stratified over [0, 1], one per pass.  The estimate
    stays unbiased, its baseline is exactly the background mean, and its
    noise at a given sample count is lower.
    """
    rng = np.random.default_rng(seed)
    n_background = len(background)
    rounds = max(1, -(-n_samples // n_background))
    n_samples = rounds * n_background
    picks = np.tile(np.arange(n_background), rounds)
    # Pass r puts each background window's position in [r, r + 1) / rounds.
    positions = (np.repeat(np.arange(rounds), n_background) + rng.random(n_samples)) / rounds
    picks_t = torch.as_tensor(picks, device=background.device)
    positions_t = torch.as_tensor(positions, dtype=row.dtype, device=row.device)

    row = row.detach()
    total = None
    for chunk in torch.split(torch.arange(n_samples, device=row.device), SHAP_GRADIENT_BATCH_SIZE):
        base = background[picks_t[chunk]].detach()
        delta = row - base
        point = (base + positions_t[chunk].reshape(-1, 1, 1) * delta).requires_grad_(True)
        outputs = wrapper(point)                                      # [batch, targets]
        n_targets = outputs.shape[1]
        if total is None:
            total = torch.zeros(*row.shape[1:], n_targets, dtype=torch.float64, device=row.device)
        for k in range(n_targets):
            (grad,) = torch.autograd.grad(outputs[:, k].sum(), point, retain_graph=k < n_targets - 1)
            total[..., k] += (grad * delta).sum(dim=0).to(torch.float64)
    return _to_numpy(total / n_samples)


def _explain_windows(wrapper, inputs: torch.Tensor, background: torch.Tensor, contexts: Sequence) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expected gradients for each window under its own fixed context.

    A window is explained on batches of interpolations between it and
    background windows.  The wrapper cannot tell which window a batch belongs
    to, so it is handed the window's context -- what must not be
    interpolated -- before each one and broadcasts it over the batch.

    Returns the SHAP values [windows, steps, features, targets], f(x) for
    every window and the mean of f over the background under that window's
    context, both [windows, targets]: the two sides of the additivity check.
    """
    per_row, fx, base = [], [], []
    # cuDNN's RNN backward requires training mode, and the models are in eval
    # mode; the native implementation has no such limit.
    with torch.backends.cudnn.flags(enabled=False):
        for j in range(len(inputs)):
            wrapper.set_row(contexts[j])
            row = inputs[j:j + 1]
            per_row.append(_expected_gradients(wrapper, row, background, SHAP_GRADIENT_NSAMPLES))
            with torch.no_grad():
                fx.append(_to_numpy(wrapper(row.detach()))[0])
                over_background = [_to_numpy(wrapper(chunk.detach())) for chunk in torch.split(background, 256)]
                base.append(np.concatenate(over_background, axis=0).mean(axis=0))
    return np.stack(per_row, axis=0), np.asarray(fx, dtype=np.float64), np.asarray(base, dtype=np.float64)


def _save_shap_arrays(plots_dir: str, model_type: str, payload: Dict) -> None:
    """Write the two .npy arrays and the self-describing cache next to them."""
    os.makedirs(plots_dir, exist_ok=True)
    np.save(os.path.join(plots_dir, f"{model_type}_shap_values_temporal.npy"), payload["temporal_shap"])
    np.save(os.path.join(plots_dir, f"{model_type}_shap_values.npy"), payload["averaged_shap"])
    cache_path = os.path.join(plots_dir, f"{model_type}_shap_cache.npz")
    np.savez_compressed(cache_path, format=np.array(SHAP_CACHE_FORMAT), **payload)
    logging.info("Saved %s SHAP arrays and cache to %s", model_type.upper(), plots_dir)


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
    return np.array(x)

def _timestep_labels(sequence_length: int, encoder_length: Optional[int] = None) -> List[str]:
    """Axis labels for the steps of a window; "t" is the last encoder step.

    With *encoder_length* shorter than the window, the steps after it are the
    decoder's and count forward from "t+1".
    """
    encoder_length = sequence_length if encoder_length is None else min(int(encoder_length), sequence_length)
    encoder = [
        "t" if steps_back == 0 else f"t-{steps_back}"
        for steps_back in range(encoder_length - 1, -1, -1)
    ]
    return encoder + [f"t+{k}" for k in range(1, sequence_length - encoder_length + 1)]


class _LSTMPredictionWrapper(torch.nn.Module):
    """Maps a window's continuous inputs to the LSTM's outputs, for SHAP.

    SHAP explains the continuous features only; the categorical codes of the
    window being explained are held fixed (``set_row``) and broadcast over
    whatever batch of interpolations it is passed.
    """

    def __init__(self, lstm_model, seq_len: int):
        super().__init__()
        self.lstm_model = lstm_model
        self.seq_len = seq_len
        self.fixed_cat = None  # [1, seq_len, num_cat]

    def set_row(self, cat_indices: torch.Tensor) -> None:
        self.fixed_cat = cat_indices

    def forward(self, x):
        batch_size = x.shape[0]
        mask = torch.ones(batch_size, self.seq_len, dtype=torch.float32, device=x.device)
        cat = self.fixed_cat.to(x.device).expand(batch_size, -1, -1)
        return self.lstm_model(x, mask=mask, cat_indices=cat)


def get_lstm_shap_values(run_id, X_test: pd.DataFrame, sequence_length=1, group_ids=None, out_dir: Optional[str] = None):
    """Expected-gradients SHAP for the LSTM over a random sample of windows.

    Every window of *X_test* that lies inside one series is a candidate;
    *group_ids* gives each row's (Model, Scenario, Region) series id, so no
    window straddles two series.  ``draw_window_sample`` picks the windows to
    explain and the background from all of them.  Arrays go to *out_dir*
    (the run's plots directory when None).
    """
    from src.trainers.lstm_trainer import LSTMModel
    logging.info("Loading LSTM model...")
    model_path = os.path.join(get_run_root(run_id), "final", "best.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model checkpoint not found: {model_path}")
    model = LSTMModel.load_from_checkpoint(model_path)
    from src.utils.run_store import RunStore
    store = RunStore(run_id)
    scaler_X = store.load_artifact("lstm_scaler_X.pkl")
    model.eval()
    features, targets = store.load_features()
    categorical_features = []
    if store.has_train_meta():
        meta = store.load_train_meta()
        features = meta.get("lstm_raw_features", features)
        categorical_features = meta.get("lstm_categorical_features", [])
    continuous_features = [f for f in features if f not in categorical_features]

    def preprocess_features(data, continuous_features, categorical_features, scaler_X, mask_value=-1.0):
        from configs.data import REGION_CATEGORIES
        from src.data.preprocess import set_region_categories
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
                if not REGION_CATEGORIES:
                    set_region_categories(data[col])
                cat_codes[col] = (
                    pd.Categorical(data[col].astype(str), categories=REGION_CATEGORIES, ordered=True)
                    .codes.astype(np.int64)
                )
            else:
                cat_codes[col] = data[col].astype('category').cat.codes.values.astype(np.int64)
        X_cat = np.column_stack([cat_codes[c] for c in categorical_features]) if categorical_features else np.empty((len(data), 0), dtype=np.int64)
        return X_cont_scaled, X_cat

    groups = None if group_ids is None else np.asarray(group_ids)
    cont, cat = preprocess_features(X_test, continuous_features, categorical_features, scaler_X)
    cont = np.asarray(cont, dtype=np.float32)

    # Every sliding window that stays inside one series.  The rows arrive
    # sorted by series, so a window whose first and last rows share a group
    # id lies within it -- the windows LSTMDataset builds per group.
    starts = np.arange(max(len(cont) - sequence_length + 1, 0))
    if groups is not None and len(starts):
        starts = starts[groups[starts] == groups[starts + sequence_length - 1]]
    explained_pos, background_pos = draw_window_sample(len(starts))
    logging.info(
        "LSTM SHAP: explaining %d of %d windows against a background of %d",
        len(explained_pos), len(starts), len(background_pos),
    )

    def windows(positions):
        rows = starts[positions][:, None] + np.arange(sequence_length)[None, :]   # [windows, seq_len]
        return torch.from_numpy(cont[rows]), torch.from_numpy(cat[rows])

    test_inputs, test_cat_seq = windows(explained_pos)
    background_data, _ = windows(background_pos)

    wrapper = _LSTMPredictionWrapper(model, sequence_length)
    wrapper.eval()
    device = next(model.parameters()).device
    background_data = background_data.to(device)
    test_inputs = test_inputs.to(device)
    test_cat_seq = test_cat_seq.to(device)
    n_targets = len(targets)
    if len(test_inputs):
        logging.info("Calculating LSTM SHAP values...")
        shap_values, fx, base_values = _explain_windows(
            wrapper, test_inputs, background_data,
            [test_cat_seq[j:j + 1] for j in range(len(test_inputs))],
        )
    else:
        shap_values = np.empty((0, sequence_length, cont.shape[1], n_targets))
        fx = base_values = np.empty((0, n_targets))
    original_temporal_shap = shap_values.copy()
    averaged = np.mean(shap_values, axis=1)
    _log_additivity("LSTM", additivity_report(original_temporal_shap, fx, base_values, targets))

    test_sequences_np = _to_numpy(test_inputs)
    last_rows = starts[explained_pos] + sequence_length - 1
    _save_shap_arrays(_plots_dir(run_id, out_dir), "lstm", {
        "temporal_shap": original_temporal_shap,
        "averaged_shap": averaged,
        "test_inputs": test_sequences_np,
        "fx": fx,
        "base_values": base_values,
        "feature_names": np.array(continuous_features, dtype=str),
        "targets": np.array(list(targets), dtype=str),
        "explained_rows": last_rows,
        "n_candidate_windows": np.array(len(starts)),
        "n_background": np.array(len(background_pos)),
        "encoder_length": np.array(sequence_length),
    })
    # The explained windows' latest rows, unscaled.
    X_processed = X_test.iloc[last_rows].loc[:, continuous_features]
    return original_temporal_shap, averaged, X_processed, test_sequences_np


class _TFTPredictionWrapper(torch.nn.Module):
    """Maps a window's continuous inputs to every target's mean forecast, for SHAP.

    SHAP needs a module taking a single tensor.  That tensor is the window's
    continuous inputs over the encoder steps followed by the decoder steps,
    ``[batch, encoder_length + decoder steps, n_continuous]``, so the decoder
    inputs -- which carry most of what the model reads -- are attributed like
    the encoder's.  Each call returns ``[batch, n_targets]``: every target's
    forecast averaged over its valid horizon.

    Everything else the TFT reads -- categorical codes, encoder and decoder
    lengths, target scale -- belongs to the one window being explained.
    ``set_row`` receives that window's batch (batch size 1) and ``forward``
    broadcasts it over the interpolations it is passed, so no window
    is scored with another's decoder context and no categorical code is
    interpolated.
    """

    # The entries SHAP's tensor replaces.
    _CONTINUOUS_KEYS = ("encoder_cont", "decoder_cont")

    def __init__(self, tft_model, encoder_length: int):
        super().__init__()
        self.tft_model = tft_model
        self.encoder_length = int(encoder_length)
        self.row_context: Optional[dict] = None

    @staticmethod
    def continuous_inputs(batch: dict) -> torch.Tensor:
        """The tensor SHAP attributes: encoder then decoder continuous inputs."""
        return torch.cat([batch["encoder_cont"], batch["decoder_cont"]], dim=1)

    def set_row(self, row_context: dict) -> None:
        """Fix the non-continuous inputs to those of one window (batch size 1)."""
        self.row_context = row_context

    def _build_batch(self, x):
        """The TFT batch dict: SHAP's continuous inputs in the row's own context."""
        if self.row_context is None:
            raise RuntimeError("set_row() must be called with the explained window's batch first")
        batch_size = x.shape[0]
        device = x.device

        def _broadcast(value):
            if isinstance(value, torch.Tensor):
                value = value.to(device)
                if value.dim() >= 1 and value.size(0) == 1:
                    return value.expand(batch_size, *value.shape[1:])
                return value
            if isinstance(value, (list, tuple)):
                return type(value)(_broadcast(item) for item in value)
            return value

        batch_dict = {
            key: _broadcast(value)
            for key, value in self.row_context.items()
            if key not in self._CONTINUOUS_KEYS
        }
        batch_dict["encoder_cont"] = x[:, :self.encoder_length]
        batch_dict["decoder_cont"] = x[:, self.encoder_length:]
        return batch_dict

    @staticmethod
    def _point_forecasts(output) -> List[torch.Tensor]:
        """Each target's point forecast, shaped [batch, horizon].

        The TFT returns a NamedTuple whose fields are, in order, ``prediction``,
        ``encoder_attention``, ``decoder_attention``, ... — and with more than
        one target ``prediction`` is a *list* of tensors.  Scanning the tuple
        for the first Tensor therefore skips it and lands on the attention
        weights, so read the field by name.
        """
        prediction = getattr(output, "prediction", None)
        if prediction is None:
            if isinstance(output, torch.Tensor):
                prediction = output
            else:
                raise ValueError(
                    f"TFT output of type {type(output)} has no 'prediction' field"
                )

        per_target = list(prediction) if isinstance(prediction, (list, tuple)) else [prediction]
        forecasts = []
        for pred in per_target:
            if pred.dim() == 3:
                # Trailing axis is the loss's output size: 1 for a point forecast,
                # n_quantiles otherwise (take the median).
                pred = pred[..., pred.shape[-1] // 2]
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            forecasts.append(pred)
        return forecasts

    def forward(self, x):
        batch_dict = self._build_batch(x)
        forecasts = self._point_forecasts(self.tft_model(batch_dict))

        # Mean over the forecast horizon, ignoring padded decoder steps.
        lengths = batch_dict.get("decoder_lengths")
        means = []
        for pred in forecasts:
            if lengths is not None and torch.is_tensor(lengths) and lengths.numel() == pred.shape[0]:
                steps = torch.arange(pred.shape[1], device=pred.device)
                valid = (steps.unsqueeze(0) < lengths.to(pred.device).reshape(-1, 1)).to(pred.dtype)
                means.append((pred * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1))
            else:
                means.append(pred.mean(dim=1))
        return torch.stack(means, dim=1)


def _load_shap_cache(cache_path: str):
    """A cache in the current layout, or None when there is none to trust."""
    if not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=True) as cached:
            if "format" not in cached.files or int(cached["format"]) != SHAP_CACHE_FORMAT:
                logging.info("Ignoring SHAP cache %s: written in an earlier layout", cache_path)
                return None
            return {name: cached[name] for name in cached.files}
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to load cached SHAP data from %s: %s", cache_path, exc)
        return None


def _slice_batch(batch: dict, j: int) -> dict:
    """Window *j* of a collated TFT batch, keeping the batch axis (size 1)."""
    def _pick(value):
        if isinstance(value, torch.Tensor):
            return value[j:j + 1]
        if isinstance(value, (list, tuple)):
            return type(value)(_pick(item) for item in value)
        return value
    return {key: _pick(value) for key, value in batch.items()}


def get_tft_shap_values(
    run_id,
    X_test: pd.DataFrame,
    *,
    use_cached: bool = True,
    out_dir: Optional[str] = None,
):
    """Expected-gradients SHAP for the TFT over a random sample of windows.

    The candidates are the windows of *X_test* with a full encoder and a full
    decoder; ``draw_window_sample`` picks the ones to explain and the
    background from all of them.  The continuous inputs of the encoder and
    the decoder steps are attributed (see ``_TFTPredictionWrapper``).  Arrays
    go to *out_dir* (the run's plots directory when None).

    Returns the SHAP values ``[windows, steps, features, targets]``, their
    mean over steps, the windows' first-step inputs as a frame, the explained
    inputs ``[windows, steps, features]`` and the feature names.
    """
    from src.trainers.tft_model import load_tft_checkpoint
    from src.trainers.tft_dataset import load_dataset_template, from_train_template
    model_path = os.path.join(get_run_root(run_id), "final", "best.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFT model checkpoint not found: {model_path}")

    from src.utils.run_store import RunStore
    store = RunStore(run_id)
    _, targets = store.load_features()

    plots_dir = _plots_dir(run_id, out_dir)
    cache_path = os.path.join(plots_dir, "tft_shap_cache.npz")

    cached = _load_shap_cache(cache_path) if use_cached else None
    if cached is not None:
        feature_names = cached["feature_names"].tolist()
        test_inputs_np = cached["test_inputs"]
        X_processed = pd.DataFrame(test_inputs_np[:, 0, :], columns=feature_names)
        logging.info("Loaded cached TFT SHAP data from %s", cache_path)
        return cached["temporal_shap"], cached["averaged_shap"], X_processed, test_inputs_np, feature_names

    logging.info("Loading TFT model...")
    model = load_tft_checkpoint(run_id)
    model.eval()

    logging.info("Loading TFT dataset template...")
    try:
        train_template = load_dataset_template(run_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No TFT dataset template for run {run_id}; the train phase saves it. "
            "Re-run the train phase before plotting SHAP."
        ) from exc

    # Only series with room for a full encoder + decoder window are explained;
    # the template carries the lengths the model was trained with.
    encoder_length = int(train_template.max_encoder_length)
    prediction_length = int(train_template.max_prediction_length)
    min_required_length = encoder_length + prediction_length

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

    # The dataset also cuts windows with a shortened decoder near the end of
    # each series.  Their padded steps are not inputs of the forecast, and a
    # background window's padding is no baseline for a real step, so only
    # full windows are candidates.
    window_lengths = test_dataset.index["sequence_length"].to_numpy()
    candidates = np.flatnonzero(window_lengths == min_required_length)
    explained_pos, background_pos = draw_window_sample(len(candidates))
    logging.info(
        "TFT SHAP: explaining %d of %d full windows (%d in all) against a background of %d",
        len(explained_pos), len(candidates), len(window_lengths), len(background_pos),
    )
    if len(explained_pos) == 0:
        raise ValueError("No full encoder + decoder window to explain.")

    def collate(positions) -> dict:
        x, _ = test_dataset._collate_fn([test_dataset[int(i)] for i in candidates[positions]])
        full = (x["encoder_lengths"] == encoder_length) & (x["decoder_lengths"] == prediction_length)
        if not bool(full.all()):
            raise ValueError("A sampled TFT window is shorter than the full encoder + decoder length.")
        return x

    explained_batch = collate(explained_pos)
    background_batch = collate(background_pos)

    features = list(test_dataset.reals)
    device = next(model.parameters()).device
    test_inputs = _TFTPredictionWrapper.continuous_inputs(explained_batch).to(device)
    background_inputs = _TFTPredictionWrapper.continuous_inputs(background_batch).to(device)
    if test_inputs.shape[-1] != len(features):
        raise ValueError(
            f"The TFT batch carries {test_inputs.shape[-1]} continuous inputs but the dataset names {len(features)}."
        )

    contexts = [_slice_batch(explained_batch, j) for j in range(len(test_inputs))]
    wrapper = _TFTPredictionWrapper(model, encoder_length)
    wrapper.eval()

    # Expected gradients rather than DeepLIFT (shap.DeepExplainer), whose
    # rescale rules do not cover the TFT's gating and attention ops, so its
    # attributions do not sum to the prediction difference they are meant to
    # decompose.
    logging.info("Calculating TFT SHAP values for %d targets...", len(targets))
    original_temporal_shap, fx, base_values = _explain_windows(wrapper, test_inputs, background_inputs, contexts)
    averaged = np.mean(original_temporal_shap, axis=1)  # [samples, features, targets]
    _log_additivity("TFT", additivity_report(original_temporal_shap, fx, base_values, targets))

    test_inputs_np = _to_numpy(test_inputs)
    decoded = test_dataset.decoded_index.iloc[candidates[explained_pos]]
    _save_shap_arrays(plots_dir, "tft", {
        "temporal_shap": original_temporal_shap,
        "averaged_shap": averaged,
        "test_inputs": test_inputs_np,
        "fx": fx,
        "base_values": base_values,
        "feature_names": np.array(features, dtype=str),
        "targets": np.array(list(targets), dtype=str),
        "explained_windows": decoded.astype(str).to_numpy(),
        "explained_window_columns": np.array(list(decoded.columns), dtype=str),
        "n_candidate_windows": np.array(len(candidates)),
        "n_background": np.array(len(background_pos)),
        "encoder_length": np.array(encoder_length),
    })

    X_processed = pd.DataFrame(test_inputs_np[:, 0, :], columns=features)
    return original_temporal_shap, averaged, X_processed, test_inputs_np, features

def plot_lstm_shap(
    run_id,
    X_test_with_index: pd.DataFrame,
    features: List[str],
    targets: List[str],
    sequence_length=1,
    region: Optional[str] = DEFAULT_REGION,
    region_series: Optional[pd.Series] = None,
    out_dir: Optional[str] = None,
):
    """SHAP plots for the LSTM, over *region* only.

    The frame's own Region column holds the integer codes the embeddings were
    fit on, so callers pass *region_series* -- the decoded labels, aligned row
    for row -- for the filter to match against.  Arrays and plots go to
    *out_dir*, or to the run's plots directory when it is None.
    """
    logging.info("Creating LSTM SHAP plots...")
    model_path = os.path.join(get_run_root(run_id), "final", "best.ckpt")
    if not os.path.exists(model_path):
        logging.warning("Skipping LSTM SHAP plots: model checkpoint not found at %s", model_path)
        return
    # Optional region filter before scenario grouping (works on index frame)
    X_filtered, _, pre_rows, post_rows, matched, _mode = filter_index_frame_by_region(
        X_test_with_index,
        region,
        region_series=region_series,
        log_prefix="Applied region filter",
    )

    if region is not None and not matched:
        # Plotting every region under a filename that says R10 is worse than
        # plotting nothing: lstm_87 shipped exactly that, from a warning buried
        # in the log, because the codes could never match a region prefix.
        logging.error(
            "Skipping LSTM SHAP plots: region filter %r matched no rows of %d. "
            "Pass region_series with decoded Region labels.",
            region, pre_rows,
        )
        return

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

    # Series contiguous and in time order, so the sliding windows below can
    # be kept inside one series by group id.
    series_cols = [c for c in INDEX_COLUMNS if c in X_joined.columns]
    if series_cols:
        sort_cols = series_cols + (["Year"] if "Year" in X_joined.columns else [])
        X_joined = X_joined.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        group_ids = X_joined.groupby(series_cols, sort=False).ngroup().to_numpy()
    else:
        group_ids = None
    X_test = X_joined.drop(columns=NON_FEATURE_COLUMNS, errors="ignore").reset_index(drop=True)

    logging.info(
        "LSTM SHAP: %d rows after region+scenario filtering (%d -> %d groups by %s)",
        X_test.shape[0],
        total_groups,
        used_groups,
        ",".join(group_cols) if group_cols else "<none>",
    )
    try:
        temporal_shap, averaged, X_proc, test_seq = get_lstm_shap_values(
            run_id, X_test, sequence_length, group_ids=group_ids, out_dir=out_dir,
        )
        # The saved arrays stay in the model's output units; the plots share
        # one scale with the other models'.
        temporal_shap = normalise_temporal_shap(temporal_shap)
        draw_shap_all_timesteps_plot(run_id, temporal_shap, test_seq, features, targets, sequence_length, model_type="lstm", out_dir=out_dir)
        draw_shap_all_timesteps_plot(run_id, temporal_shap, test_seq, features, targets, sequence_length, model_type="lstm", xlim_range=(-0.3, 0.5), out_dir=out_dir)
        draw_temporal_shap_plot(run_id, temporal_shap, features, targets, sequence_length, model_type="lstm", out_dir=out_dir)
    except Exception as e:
        logging.error("Failed to create LSTM SHAP plots: %s", e)
        logging.exception("Full error traceback:")

def plot_tft_shap(
    run_id,
    X_test_with_index: pd.DataFrame,
    features: List[str],
    targets: List[str],
    region: Optional[str] = DEFAULT_REGION,
    use_cached: bool = True,
    out_dir: Optional[str] = None,
):
    """SHAP plots for the TFT, over *region* only.

    Arrays and plots go to *out_dir*, or to the run's plots directory when it
    is None.
    """
    logging.info("Creating TFT SHAP plots...")
    model_path = os.path.join(get_run_root(run_id), "final", "best.ckpt")
    if not os.path.exists(model_path):
        logging.warning("Skipping TFT SHAP plots: model checkpoint not found at %s", model_path)
        return

    # For TFT, we need to keep grouping columns for sequence filtering
    # Keep all columns that TFT needs: features, targets, group_ids, categorical columns, time_idx
    from configs.models.tft import TFTDatasetConfig

    config = TFTDatasetConfig()
    # Include all columns that TFT needs: features, targets, group_ids, categoricals, time_idx, and time-varying columns
    time_known = ["Year", "DeltaYears"]  # From TFT config
    from src.data.preprocess import observed_mask_columns
    obs_cols = observed_mask_columns(targets)
    required_columns = set(features + targets + config.group_ids + CATEGORICAL_COLUMNS + [config.time_idx] + time_known + obs_cols)
    # Optional region filter prior to scenario sampling (robust, on index frame)
    X_filtered, _, pre_rows, post_rows, matched, _mode = filter_index_frame_by_region(
        X_test_with_index,
        region,
        log_prefix="Applied region filter",
    )

    if region is not None and not matched:
        # Same guard as the LSTM: plotting every region under a filename that
        # names one is worse than plotting nothing.
        logging.error(
            "Skipping TFT SHAP plots: region filter %r matched no rows of %d.",
            region, pre_rows,
        )
        return

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

    try:
        temporal_shap, averaged, X_proc, test_seq, feature_names = get_tft_shap_values(
            run_id, X_test, use_cached=use_cached, out_dir=out_dir,
        )
        # Encoder steps followed by decoder steps.
        sequence_length = test_seq.shape[1]
        from src.trainers.tft_dataset import load_dataset_template
        encoder_length = int(load_dataset_template(run_id).max_encoder_length)
        # The saved arrays stay in the model's output units; the plots share
        # one scale with the other models'.
        temporal_shap = normalise_temporal_shap(temporal_shap)
        draw_shap_all_timesteps_plot(run_id, temporal_shap, test_seq, feature_names, targets, sequence_length, model_type="tft", out_dir=out_dir)
        draw_shap_all_timesteps_plot(run_id, temporal_shap, test_seq, feature_names, targets, sequence_length, model_type="tft", xlim_range=(-0.3, 0.5), out_dir=out_dir)
        draw_temporal_shap_plot(run_id, temporal_shap, feature_names, targets, sequence_length, model_type="tft", encoder_length=encoder_length, out_dir=out_dir)
    except Exception as e:
        logging.error("Failed to create TFT SHAP plots: %s", e)
        logging.exception("Full error traceback:")

def draw_shap_all_timesteps_plot(run_id: str, temporal_shap_values, test_sequences_np, features: List[str], targets: List[str], sequence_length: int, model_type: str = "lstm", xlim_range: Optional[tuple] = None, out_dir: Optional[str] = None) -> None:
    """Beeswarm per target of SHAP summed over the steps, coloured by the mean input.

    *temporal_shap_values* is expected on the shared scale
    (``normalise_temporal_shap``), which is what the x-axis label states.
    """
    if sequence_length <= 0:
        logging.warning("Invalid sequence_length=%d; skipping all-timesteps SHAP plot", sequence_length)
        return
    feature_count = temporal_shap_values.shape[2] if temporal_shap_values.ndim >= 3 else 0
    features = _align_feature_names(features, feature_count)

    # Aggregate SHAP values across timesteps: sum the signed contributions per feature
    # temporal_shap_values shape: [samples, timesteps, features, targets]
    # test_sequences_np shape: [samples, timesteps, features]
    # After aggregation: [samples, features] — one row per feature in beeswarm
    shap_agg = np.sum(temporal_shap_values, axis=1)  # [samples, features, targets]
    X_agg = np.mean(test_sequences_np, axis=1)        # [samples, features] — mean feature value for color

    display_names = build_feature_display_names(features)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': SHAP_FONT_SIZE})
    num_targets = len(targets)
    fig, axes = make_grid(num_targets, base_figsize=SHAP_GRID_FIGSIZE)

    # Create directory for individual plots
    plots_dir = _plots_dir(run_id, out_dir)
    indiv_plots_dir = os.path.join(plots_dir, 'indiv_plots', 'shap')
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
                max_display=SHAP_MAX_DISPLAY,
                xlim_range=xlim_range,
            )
            fig_local.tight_layout()
        render_external_plot(ax, _plot)
        ax.set_title(f"Impact on {targets[i]} ({output_unit(targets[i])})")

        # Save individual plot for this target
        fig_indiv = plt.figure(figsize=SHAP_INDIVIDUAL_FIGSIZE)
        ax_indiv = fig_indiv.add_subplot(111)
        draw_shap_beeswarm(
            ax_indiv,
            shap_agg[:, :, i],
            X_agg,
            display_names,
            max_display=SHAP_MAX_DISPLAY,
            xlim_range=xlim_range,
            title=f"Impact on {targets[i]} ({output_unit(targets[i])})",
        )
        fig_indiv.tight_layout()
        indiv_filename = f'{targets[i]}_match_xgb_range.png' if xlim_range is not None else f'{targets[i]}.png'
        fig_indiv.savefig(os.path.join(indiv_plots_dir, indiv_filename), dpi=300, bbox_inches='tight')
        plt.close(fig_indiv)

    fig.tight_layout()
    filename = f'{model_type}_shap_plot_match_xgb_range.png' if xlim_range is not None else f'{model_type}_shap_plot.png'
    fig.savefig(os.path.join(plots_dir, filename))
    plt.close(fig)

def draw_temporal_shap_plot(run_id: str, temporal_shap_values, features: List[str], targets: List[str], sequence_length: int, model_type: str = "lstm", encoder_length: Optional[int] = None, out_dir: Optional[str] = None) -> None:
    """Heatmap of mean |SHAP| per (step, feature) for each target's top features.

    The steps are encoder steps unless *encoder_length* is shorter than
    *sequence_length*; the ones after it are then the decoder's.
    """
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    plt.rcParams.update({'font.size': SHAP_FONT_SIZE})
    num_targets = len(targets)
    fig, axes = make_grid(num_targets, base_figsize=SHAP_GRID_FIGSIZE)
    feature_count = temporal_shap_values.shape[2] if temporal_shap_values.ndim >= 3 else 0
    features = _align_feature_names(features, feature_count)
    step_labels = _timestep_labels(sequence_length, encoder_length)
    step_axis = "Encoder step" if encoder_length is None or encoder_length >= sequence_length else "Step (encoder up to t, decoder after)"
    for i, ax in tqdm(enumerate(axes), total=num_targets, desc="Creating temporal SHAP plots"):
        if i >= num_targets:
            ax.axis('off')
            continue
        target_shap = temporal_shap_values[:, :, :, i]
        avg_importance = np.mean(np.abs(target_shap), axis=(0, 1))
        top_idx = np.argsort(avg_importance)[-SHAP_MAX_DISPLAY:][::-1]
        time_importance = np.mean(np.abs(target_shap[:, :, top_idx]), axis=0)
        im = ax.imshow(time_importance.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xticks(range(sequence_length))
        ax.set_xticklabels(step_labels)
        display_names = build_feature_display_names([features[idx] for idx in top_idx])
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([n[:30] + '...' if len(n) > 30 else n for n in display_names], fontsize=10)
        ax.set_title(f"Temporal SHAP: {targets[i]} ({output_unit(targets[i])})", fontsize=14)
        ax.set_xlabel(step_axis, fontsize=12)
        ax.set_ylabel("Features", fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    os.makedirs(_plots_dir(run_id, out_dir), exist_ok=True)
    fig.savefig(os.path.join(_plots_dir(run_id, out_dir), f'{model_type}_temporal_shap_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    create_timestep_comparison_plots(run_id, temporal_shap_values, features, targets, sequence_length, model_type, encoder_length=encoder_length, out_dir=out_dir)


def create_timestep_comparison_plots(run_id: str, temporal_shap_values, features: List[str], targets: List[str], sequence_length: int, model_type: str = "lstm", encoder_length: Optional[int] = None, out_dir: Optional[str] = None) -> None:
    """Bar chart of mean |SHAP| per step for each target."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': SHAP_FONT_SIZE})
    fig, axes = make_grid(len(targets), base_figsize=SHAP_GRID_FIGSIZE)
    labels = _timestep_labels(sequence_length, encoder_length)
    step_axis = "Encoder step" if encoder_length is None or encoder_length >= sequence_length else "Step (encoder up to t, decoder after)"
    colors = matplotlib.colormaps['viridis'](np.linspace(0, 1, sequence_length))
    for i, ax in enumerate(axes):
        if i >= len(targets):
            ax.axis('off')
            continue
        target_shap = temporal_shap_values[:, :, :, i]
        timestep_importance = np.mean(np.abs(target_shap), axis=(0, 2))
        bars = ax.bar(range(sequence_length), timestep_importance, color=colors, alpha=0.8)
        ax.set_title(f"Time Step Importance: {targets[i]} ({output_unit(targets[i])})", fontsize=14)
        ax.set_xlabel(step_axis, fontsize=12)
        ax.set_ylabel("Average |SHAP| Value", fontsize=12)
        ax.set_xticks(range(sequence_length))
        ax.set_xticklabels(labels)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + h*0.01, f"{h:.3f}", ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    os.makedirs(_plots_dir(run_id, out_dir), exist_ok=True)
    fig.savefig(os.path.join(_plots_dir(run_id, out_dir), f'{model_type}_timestep_importance.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info("Temporal SHAP plots saved")
