"""Evaluation utilities.

Note: pytorch-forecasting 1.4.x returns normalized outputs in Prediction.output;
original-scale values may not be present unless Prediction.prediction is
populated. The TFT prediction path enforces original-scale predictions so
metrics are never computed on mixed scales — see predict_tft in
src/trainers/tft_trainer.py.
"""

from sklearn.metrics import mean_squared_error
import os
import logging
import time
import numpy as np
import pandas as pd
import concurrent.futures

from typing import Optional

from configs.data import INDEX_COLUMNS, NON_FEATURE_COLUMNS, N_LAG_FEATURES
from src.utils.regions import SCALE_ORDER_COARSEST_FIRST, scale_of_frame
from src.utils.utils import format_number, get_run_root

# How often a long-running loop reports progress to the run log.
PROGRESS_LOG_SECONDS = 60

def group_test_data(X_test_with_index, cache=None):
    """Split the test frame into per-group index lists and feature matrices.

    *cache* holds results across search trials, which re-group the same
    validation frame for every hyperparameter configuration.
    """
    if cache is None:
        cache = {}

    cache_key = (id(X_test_with_index), X_test_with_index.shape, tuple(INDEX_COLUMNS), tuple(NON_FEATURE_COLUMNS))

    cached = cache.get(cache_key)
    if cached is not None:
        cached_frame, result = cached
        # id() is only unique among live objects, so confirm identity rather
        # than serving another frame that landed on a recycled address.
        if cached_frame is X_test_with_index:
            return result

    feature_columns = X_test_with_index.drop(columns=NON_FEATURE_COLUMNS, errors='ignore').columns
    grouped = X_test_with_index.groupby(INDEX_COLUMNS, sort=False)

    group_indices_list = []
    group_matrices = []
    for group_key in grouped.groups:
        group_df = grouped.get_group(group_key)
        group_indices_list.append(group_df.index.tolist())
        group_matrices.append(group_df[feature_columns].to_numpy())

    result = (group_indices_list, group_matrices)
    # Holding the frame also stops its id being reused while cached.
    cache[cache_key] = (X_test_with_index, result)

    return result


def autoregressive_predictions(model, group_indices, group_matrix, start_pos, y_scaler=None, x_scaler=None, feature_columns=None):
    """
    Generate autoregressive predictions for a single grouped series.

    Notes:
    - Supports arbitrary N_LAG_FEATURES based on configs.data.N_LAG_FEATURES.
    - Locates lagged feature columns by name: prev_<var> or prev{lag}_<var>.
    - Assumes model.predict returns a vector of targets aligned with
      OUTPUT_VARIABLES[:num_targets].
    """
    # Infer number of targets from a single prediction
    first_pred = model.predict(group_matrix[start_pos:start_pos + 1])
    # Ensure 2D shape (1, num_targets)
    if first_pred.ndim == 1:
        num_targets = first_pred.shape[0]
    else:
        num_targets = first_pred.shape[1]
    preds_target = np.full((len(group_indices), num_targets), np.nan, dtype=float)

    # Seed with initial prediction
    preds_target[start_pos, :] = first_pred.reshape(1, -1)[0]

    # Precompute lagged feature column indices once
    from configs.data import OUTPUT_VARIABLES
    out_vars = OUTPUT_VARIABLES[:num_targets]
    # Build a dict: lag -> list of column indices (or None) for each output variable
    lag_col_indices = {}
    if feature_columns is None:
        # If not provided, derive from matrix width assuming caller aligned order with X_test_with_index
        feature_columns = []
    for lag in range(1, N_LAG_FEATURES + 1):
        cols_for_lag = []
        for var in out_vars:
            col_name = (f"prev_{var}" if lag == 1 else f"prev{lag}_{var}")
            try:
                col_idx = feature_columns.index(col_name)
            except (ValueError, AttributeError):
                col_idx = None
            cols_for_lag.append(col_idx)
        lag_col_indices[lag] = cols_for_lag

    # If scalers are provided, precompute mean/scale for targets and lagged feature columns
    x_means_attr = getattr(x_scaler, 'mean_', None) if x_scaler is not None else None
    x_scales_attr = getattr(x_scaler, 'scale_', None) if x_scaler is not None else None
    y_means_attr = getattr(y_scaler, 'mean_', None) if y_scaler is not None else None
    y_scales_attr = getattr(y_scaler, 'scale_', None) if y_scaler is not None else None
    def _safe_len(x):
        try:
            return len(x)
        except Exception:
            return -1
    use_scalers = (
        isinstance(x_means_attr, (list, np.ndarray)) and isinstance(x_scales_attr, (list, np.ndarray)) and
        isinstance(y_means_attr, (list, np.ndarray)) and isinstance(y_scales_attr, (list, np.ndarray)) and
        _safe_len(x_means_attr) == _safe_len(feature_columns)
    )
    if not use_scalers:
        if x_scaler is not None or y_scaler is not None:
            logging.warning(
                "Scalers provided but unusable (x_scaler.mean_ length %d != feature_columns length %d); "
                "lag updates will insert y-scaled values into x-scaled columns",
                _safe_len(x_means_attr), _safe_len(feature_columns),
            )
        else:
            logging.debug("No scalers provided; lag updates will insert predictions as-is")
    if use_scalers:
        y_means = np.asarray(y_means_attr)[:num_targets]
        y_scales = np.asarray(y_scales_attr)[:num_targets]
        # For each lag, align x_scaler stats to lag columns
        lag_x_means = {}
        lag_x_scales = {}
        x_means_full = np.asarray(x_means_attr)
        x_scales_full = np.asarray(x_scales_attr)
        for lag, cols in lag_col_indices.items():
            lag_x_means[lag] = np.array([(x_means_full[c] if c is not None else 0.0) for c in cols])
            lag_x_scales[lag] = np.array([(x_scales_full[c] if c is not None else 1.0) for c in cols])

    # Roll forward autoregressively
    for t in range(start_pos + 1, len(group_indices)):
        X_test_curr = group_matrix[t].copy()

        # Update lagged features using previous predictions
        for lag in range(1, N_LAG_FEATURES + 1):
            src_t = t - lag
            if src_t < start_pos:
                continue
            for pred_idx in range(num_targets):
                col_idx = lag_col_indices[lag][pred_idx]
                if col_idx is None:
                    continue
                y_val_scaled = preds_target[src_t, pred_idx]
                if use_scalers:
                    raw_val = y_val_scaled * y_scales[pred_idx] + y_means[pred_idx]
                    x_mean = lag_x_means[lag][pred_idx]
                    x_scale = lag_x_scales[lag][pred_idx]
                    X_test_curr[col_idx] = (raw_val - x_mean) / x_scale
                else:
                    X_test_curr[col_idx] = y_val_scaled

        # Predict for all targets at time t
        next_pred = model.predict(X_test_curr.reshape(1, -1))
        preds_target[t, :] = next_pred.reshape(1, -1)[0]

    return preds_target


def test_xgb_autoregressively(
    X_test_with_index,
    y_test,
    run_id=None,
    model=None,
    disable_progress: bool = False,
    cache=None,
    y_scaler=None,
    x_scaler=None,
    max_workers: Optional[int] = None,
):
    """Test the model autoregressively on the test set.

    *disable_progress* silences the periodic progress line and the closing
    RMSE; the search sets it, where one line per trial is enough and these
    would arrive once per fold.
    """
    if cache is None:
        cache = {}
    
    # Load scalers if not provided and run_id is available
    if y_scaler is None and x_scaler is None and run_id is not None:
        try:
            from src.utils.run_store import RunStore
            store = RunStore(run_id)
            y_scaler = store.load_artifact("y_scaler.pkl")
            x_scaler = store.load_artifact("x_scaler.pkl")
        except Exception:
            logging.warning("Could not load scalers, falling back to original behavior")
            y_scaler = None
            x_scaler = None
        
    # group_test_data returns (group_indices_list, group_matrices, [optional lag masks...])
    group_indices_list, group_matrices = group_test_data(X_test_with_index, cache)
    full_preds = np.full(y_test.shape, np.nan, dtype=float)
    
    if model is None:
        if run_id is None:
            raise ValueError("Either provide a preloaded `model` or a valid `run_id` to load from disk.")
        from configs.data import OUTPUT_VARIABLES
        from src.trainers.xgb_trainer import load_final_xgb_model

        n_targets = y_test.shape[1] if y_test.ndim > 1 else 1
        model = load_final_xgb_model(run_id, OUTPUT_VARIABLES[:n_targets])

    # Get feature column names
    feature_columns = [col for col in X_test_with_index.columns if col not in NON_FEATURE_COLUMNS]

    def process_group(args):
        group_indices, group_matrix = args
        # With no nan values in y_test, we always use the first instance as seed.
        start_pos = 0
        preds_target = autoregressive_predictions(model, group_indices, group_matrix, start_pos, y_scaler, x_scaler, feature_columns)
        return group_indices, preds_target

    index_to_pos = {idx: pos for pos, idx in enumerate(X_test_with_index.index)}
    groups = list(zip(group_indices_list, group_matrices))
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for group in groups:
            futures.append(executor.submit(process_group, group))
            
        total = len(futures)
        completed = 0
        last_logged = time.monotonic()

        for future in concurrent.futures.as_completed(futures):
            group_indices, preds_target = future.result()
            pos = [index_to_pos[idx] for idx in group_indices]
            full_preds[pos, :] = preds_target

            completed += 1
            # Reported to the log rather than a progress bar: the bar reached
            # only a terminal, so train.log had nothing for the six minutes
            # this takes, while its redraws filled the console log instead.
            now = time.monotonic()
            if not disable_progress and (
                completed == total or now - last_logged >= PROGRESS_LOG_SECONDS
            ):
                logging.info("Autoregressive prediction: %d/%d groups", completed, total)
                last_logged = now

    if not disable_progress:
        # y_test carries NaN at unobserved targets; score the observed ones.
        finite = np.isfinite(np.asarray(y_test, dtype=float)) & np.isfinite(full_preds)
        if finite.any():
            mse = mean_squared_error(
                np.asarray(y_test, dtype=float)[finite], full_preds[finite]
            )
            # Named for its units: this runs on the scaled targets the model
            # predicts, while save_metrics reports RMSE in absolute units a few
            # lines later, and the two differ by eight orders of magnitude.
            logging.info(
                "Autoregressive test RMSE (scaled units): %s",
                format_number(np.sqrt(mse)),
            )
        else:
            logging.warning("No observed test targets to score")

    return full_preds


def _r2(yt, yp):
    """Coefficient of determination for a single flat array pair."""
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def compute_r2_summary(y_true, y_pred, observed_mask=None) -> dict:
    """Pooled and per-target-average R2/RMSE for one (y_true, y_pred) pair.

    Mirrors the "Overall" row logic in save_metrics(), factored out so callers
    that need a quick split-level diagnostic (e.g. train/val/test R2
    breakdowns) don't have to run the full save_metrics pipeline.
    """
    yt_2d = np.asarray(y_true)
    yp_2d = np.asarray(y_pred)
    if yt_2d.ndim == 1:
        yt_2d = yt_2d.reshape(-1, 1)
        yp_2d = yp_2d.reshape(-1, 1)
    obs_2d = np.asarray(observed_mask) if observed_mask is not None else None

    target_r2s = []
    for col in range(yt_2d.shape[1]):
        yt_col, yp_col = yt_2d[:, col], yp_2d[:, col]
        if obs_2d is not None:
            col_mask = obs_2d[:, col].astype(bool)
            yt_col, yp_col = yt_col[col_mask], yp_col[col_mask]
        valid = np.isfinite(yt_col) & np.isfinite(yp_col)
        if valid.any():
            target_r2s.append(_r2(yt_col[valid], yp_col[valid]))
    per_target_r2 = float(np.mean(target_r2s)) if target_r2s else np.nan

    yt_flat, yp_flat = yt_2d.flatten(), yp_2d.flatten()
    if obs_2d is not None:
        mask = obs_2d.astype(bool).flatten()
        yt_flat, yp_flat = yt_flat[mask], yp_flat[mask]
    valid = np.isfinite(yt_flat) & np.isfinite(yp_flat)
    yt_flat, yp_flat = yt_flat[valid], yp_flat[valid]

    if len(yt_flat) == 0:
        return {
            "R2 (per-target avg)": np.nan, "R2 (pooled)": np.nan,
            "RMSE": np.nan, "MAE": np.nan, "Sample Size": 0,
        }
    return {
        "R2 (per-target avg)": per_target_r2,
        "R2 (pooled)": _r2(yt_flat, yp_flat),
        "RMSE": float(np.sqrt(mean_squared_error(yt_flat, yp_flat))),
        "MAE": float(np.mean(np.abs(yt_flat - yp_flat))),
        "Sample Size": int(len(yt_flat)),
    }


def per_target_r2_table(y_true, y_pred, targets, observed_mask=None) -> pd.DataFrame:
    """Per-output-variable R2/RMSE table (rows = targets), for ablation-style reporting."""
    yt_2d = np.asarray(y_true)
    yp_2d = np.asarray(y_pred)
    if yt_2d.ndim == 1:
        yt_2d = yt_2d.reshape(-1, 1)
        yp_2d = yp_2d.reshape(-1, 1)
    obs_2d = np.asarray(observed_mask) if observed_mask is not None else None

    rows = []
    for col, target in enumerate(targets):
        yt_col, yp_col = yt_2d[:, col], yp_2d[:, col]
        if obs_2d is not None:
            col_mask = obs_2d[:, col].astype(bool)
            yt_col, yp_col = yt_col[col_mask], yp_col[col_mask]
        valid = np.isfinite(yt_col) & np.isfinite(yp_col)
        yt_col, yp_col = yt_col[valid], yp_col[valid]
        if len(yt_col) == 0:
            rows.append({"Output Variable": target, "R2": np.nan, "RMSE": np.nan, "Sample Size": 0})
            continue
        rows.append({
            "Output Variable": target,
            "R2": _r2(yt_col, yp_col),
            "RMSE": float(np.sqrt(mean_squared_error(yt_col, yp_col))),
            "Sample Size": int(len(yt_col)),
        })
    return pd.DataFrame(rows)


def _metrics_line(headline) -> str:
    """The metric values as one log line, defined once so that the per-scale
    and overall lines cannot drift apart.

    Absolute targets run to 1e16, where %.4f prints seventeen digits of noise,
    while the scaled losses elsewhere are ~0.05; format_number picks the
    notation that stays readable across both.
    """
    return " ".join(
        f"{label}={format_number(headline[key])}"
        for label, key in (
            ("MSE", "Mean Squared Error"),
            ("RMSE", "RMSE"),
            ("MAE", "MAE"),
            ("R2_avg", "R2 Score (per-target avg)"),
            ("R2_pooled", "R2 Score (pooled)"),
            ("Pearson", "Pearson Correlation"),
        )
    )


def save_metrics(run_id, y_true, y_pred, test_data=None, observed_mask=None,
                 metrics_filename="performance.csv"):
    """Save performance metrics to a CSV file under the specified run directory.

    When *observed_mask* is provided (KEEP_PARTIAL_TARGETS=True), metrics are
    computed on observed elements only.  Otherwise all elements are used.

    *test_data* is the frame whose rows correspond one-for-one, in order, with
    the rows of *y_true*: for the sequence models that is the forecast horizon
    frame rather than the full test split.  Given it, metrics are additionally
    broken down by region scale, which is the comparison the models are read
    against each other on.

    *metrics_filename* lets callers write split-specific metrics (e.g.
    "performance_train.csv") without overwriting the canonical test-set
    "performance.csv".
    """
    if test_data is not None and len(test_data) != len(y_true):
        raise ValueError(
            f"save_metrics got {len(test_data)} frame rows for {len(y_true)} scored rows; "
            "pass the frame the predictions were made on, row for row."
        )

    def compute_metrics(y_true_subset, y_pred_subset, subset_name="Overall", obs=None):
        results = []

        def _metrics(yt, yp, per_target_r2):
            mse = mean_squared_error(yt, yp)
            mae = float(np.mean(np.abs(yt - yp)))
            rmse = float(np.sqrt(mse))
            r2_pooled = _r2(yt, yp)
            try:
                yt_f, yp_f = yt.flatten(), yp.flatten()
                pearson_corr = float(np.corrcoef(yt_f, yp_f)[0, 1]) if len(yt_f) > 1 else np.nan
            except Exception:
                pearson_corr = np.nan
            return {
                "Run ID": run_id,
                "Region Type": subset_name,
                "Mean Squared Error": mse,
                "Pearson Correlation": pearson_corr,
                "R2 Score (per-target avg)": per_target_r2,
                "R2 Score (pooled)": r2_pooled,
                "MAE": mae,
                "RMSE": rmse,
                "Sample Size": len(yt),
            }

        yt_2d = np.asarray(y_true_subset)
        yp_2d = np.asarray(y_pred_subset)
        if yt_2d.ndim == 1:
            yt_2d = yt_2d.reshape(-1, 1)
            yp_2d = yp_2d.reshape(-1, 1)
        obs_2d = np.asarray(obs) if obs is not None else None

        # Compute per-target R²
        target_r2s = []
        for col in range(yt_2d.shape[1]):
            yt_col = yt_2d[:, col]
            yp_col = yp_2d[:, col]
            if obs_2d is not None:
                col_mask = obs_2d[:, col].astype(bool)
                yt_col = yt_col[col_mask]
                yp_col = yp_col[col_mask]
            valid = np.isfinite(yt_col) & np.isfinite(yp_col)
            if valid.any():
                target_r2s.append(_r2(yt_col[valid], yp_col[valid]))
        per_target_r2 = float(np.mean(target_r2s)) if target_r2s else np.nan

        # Flatten for pooled metrics
        yt_flat = yt_2d.flatten()
        yp_flat = yp_2d.flatten()

        if obs_2d is not None:
            mask = obs_2d.astype(bool).flatten()
            yt_flat = yt_flat[mask]
            yp_flat = yp_flat[mask]

        valid = np.isfinite(yt_flat) & np.isfinite(yp_flat)
        if valid.any():
            results.append(_metrics(yt_flat[valid], yp_flat[valid], per_target_r2))

        return results

    # Compute overall metrics
    all_metrics = compute_metrics(y_true, y_pred, "Overall", obs=observed_mask)

    # If test_data is provided, compute metrics by region scale
    scales = scale_of_frame(test_data) if test_data is not None else None
    if scales is not None:
        for region_type in SCALE_ORDER_COARSEST_FIRST:
            region_positions = np.where((scales == region_type).to_numpy())[0]
            if len(region_positions) > 0:
                y_true_region = y_true[region_positions]
                y_pred_region = y_pred[region_positions]
                obs_region = observed_mask[region_positions] if observed_mask is not None else None

                if len(y_true_region) > 0:
                    region_results = compute_metrics(y_true_region, y_pred_region, region_type, obs=obs_region)
                    all_metrics.extend(region_results)

                    if region_results:
                        headline = region_results[0]
                        logging.info(
                            "Run %s %s regions (%d samples) -> %s",
                            run_id, region_type, int(headline["Sample Size"]),
                            _metrics_line(headline),
                        )

    metrics = pd.DataFrame(all_metrics)

    metrics_dir = os.path.join(get_run_root(run_id), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, metrics_filename)
    metrics.to_csv(metrics_file, index=False)
    logging.info("Metrics saved to %s.", metrics_file)

    if all_metrics:
        headline = all_metrics[0]
        logging.info(
            "Run %s overall metrics (%d samples) -> %s",
            run_id, int(headline["Sample Size"]), _metrics_line(headline),
        )
