"""TFT trainer with main orchestration functions."""

import logging
import os
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterSampler

from configs.paths import RESULTS_PATH
from configs.models import TFTSearchSpace, TFTTrainerConfig
from .tft_dataset import (
    build_datasets,
    create_combined_dataset,
    from_train_template,
    load_dataset_template,
    save_dataset_template,
)
from .tft_model import (
    create_dataloaders,
    create_final_trainer,
    create_search_trainer,
    create_tft_model,
    create_trial_checkpoint,
    load_tft_checkpoint,
)
from .tft_utils import single_gpu_env, teardown_distributed, get_default_num_workers


def hyperparameter_search_tft(
    train_dataset,
    val_dataset,
    targets: List[str],
    run_id: str,
) -> Dict[str, Any]:
    """Perform hyperparameter search for TFT model."""
    search_cfg = TFTSearchSpace()
    trainer_cfg = TFTTrainerConfig()

    search_results = []
    best_score = float("inf")
    best_params: Optional[Dict[str, Any]] = None

    for i, params in enumerate(ParameterSampler(search_cfg.param_dist, n_iter=search_cfg.search_iter_n, random_state=0)):
        logging.info(f"TFT Search Iteration {i+1}/{search_cfg.search_iter_n} - Params: {params}")

        n_targets = len(targets)
        tft = create_tft_model(train_dataset, params, n_targets)

        checkpoint_callback = create_trial_checkpoint(run_id, i)
        trainer = create_search_trainer(trainer_cfg, checkpoint_callback)

        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, trainer_cfg.batch_size)
        trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        val_loss = trainer.callback_metrics["val_loss"].item()
        search_results.append({**params, "val_loss": val_loss})

        if val_loss < best_score:
            best_score = val_loss
            best_params = params

    if best_params is None:
        raise RuntimeError("Hyperparameter search completed without selecting best_params (no iterations?).")
    logging.info(f"Best TFT Params: {best_params} with Val Loss: {best_score:.4f}")
    return best_params


def train_final_tft(
    train_dataset,
    val_dataset,
    targets: List[str],
    run_id: str,
    best_params: Dict,
    session_state: Optional[Dict] = None,
) -> None:
    """Train final TFT on combined train+val and save checkpoint."""
    trainer_cfg = TFTTrainerConfig()

    final_dir = os.path.join(RESULTS_PATH, run_id, "final")
    os.makedirs(final_dir, exist_ok=True)
    final_ckpt_path = os.path.join(final_dir, "best.ckpt")

    # Create model with best params
    n_targets = len(targets)
    tft_final = create_tft_model(train_dataset, best_params, n_targets)

    # Get original DataFrames from session_state
    train_df = None
    val_df = None
    if session_state is not None:
        train_df = session_state.get("train_data")
        val_df = session_state.get("val_data")

    if not isinstance(train_df, pd.DataFrame) or not isinstance(val_df, pd.DataFrame):
        raise RuntimeError(
            "train_final_tft requires session_state to include both train_data and val_data as DataFrames"
        )

    # Create combined dataset
    combined_dataset = create_combined_dataset(train_dataset, train_df, val_df)

    # Save dataset template
    save_dataset_template(combined_dataset, run_id)

    # Dynamic worker selection (previous refactor hard-coded 4 which can halve throughput
    # on larger CPU boxes, explaining longer wall-clock time vs older version that scaled).
    worker_count = get_default_num_workers()
    logging.info("TFT final training: using num_workers=%d (batch_size=%d, encoder_len=%d, pred_len=%d)",
                 worker_count, trainer_cfg.batch_size, train_dataset.max_encoder_length, train_dataset.max_prediction_length)

    import time as _time
    build_start = _time.time()
    combined_loader = combined_dataset.to_dataloader(
        train=True,
        batch_size=trainer_cfg.batch_size,
        num_workers=worker_count,
        persistent_workers=True,
    )
    logging.info("TFT final training: dataloader build time = %.2fs", _time.time() - build_start)

    final_trainer = create_final_trainer(trainer_cfg)
    fit_start = _time.time()
    final_trainer.fit(model=tft_final, train_dataloaders=combined_loader)
    logging.info("TFT final training: fit time = %.2fs", _time.time() - fit_start)
    final_trainer.save_checkpoint(final_ckpt_path)

    # Persist targets used for this run to prevent future mismatches
    try:
        import json as _json
        with open(os.path.join(final_dir, "targets.json"), "w") as f:
            _json.dump(targets, f)
        logging.info("Saved training targets to %s", os.path.join(final_dir, "targets.json"))
    except Exception as _e:
        logging.warning("Failed to persist training targets: %s", _e)


def predict_tft(session_state: Dict, run_id: str) -> np.ndarray:
    """Predict on forecast horizon using original-scale outputs only.

    - Builds predict dataset from saved template (deterministic ordering).
    - Requires `Prediction.prediction` (original scale). No fallback to normalized outputs.
    - Aligns multi-step horizon indices and computes metrics on valid rows only.
    """
    from src.trainers.evaluation import save_metrics

    test_data = session_state["test_data"]
    session_targets = session_state["targets"]

    with single_gpu_env():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            teardown_distributed()
            if torch.distributed.is_initialized():
                raise RuntimeError("Failed to teardown existing distributed process group before prediction.")

        model = load_tft_checkpoint(run_id)
        logging.info("Building test dataset for TFT prediction using saved template...")
        train_template = load_dataset_template(run_id)
        try:
            test_dataset = from_train_template(train_template, test_data, mode="predict")
        except Exception as e:
            raise RuntimeError(
                "Failed to build test dataset from saved template. Ensure column/dtype schema matches: "
                f"{e}"
            )

        time_idx_name = getattr(train_template, "time_idx", None)
        group_id_fields = list(getattr(train_template, "group_ids", []))
        template_targets = getattr(train_template, "target", None)
        logging.info(
            "TFT template info: time_idx=%s group_ids=%s template_targets=%s session_targets=%s",
            time_idx_name, group_id_fields, template_targets, session_targets
        )
        if not time_idx_name or not group_id_fields:
            raise ValueError("Dataset template missing time_idx or group_ids")
        # Use template-defined target order
        if template_targets is None:
            raise ValueError("Dataset template missing target list")
        if isinstance(template_targets, str):
            template_targets = [template_targets]
        targets = list(template_targets)

        # Load persisted training targets (if present) to validate alignment
        trained_targets_path = os.path.join(RESULTS_PATH, run_id, "final", "targets.json")
        trained_targets = None
        if os.path.exists(trained_targets_path):
            try:
                import json as _json
                with open(trained_targets_path, "r") as f:
                    trained_targets = _json.load(f)
                if isinstance(trained_targets, str):
                    trained_targets = [trained_targets]
                logging.info("Loaded trained targets (%d): %s", len(trained_targets), trained_targets)
            except Exception as _e:
                logging.warning("Could not read trained targets at %s: %s", trained_targets_path, _e)

        # DataLoader (use dynamic workers for speed)
        worker_count = get_default_num_workers()
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=64,
            num_workers=worker_count,
            persistent_workers=True,
        )
        logging.info("TFT predict: using num_workers=%d for inference", worker_count)

        # Log library versions to precisely diagnose behavior differences
        try:
            import pytorch_forecasting as pf  # type: ignore
            pf_ver = getattr(pf, "__version__", "unknown")
        except Exception:
            pf_ver = "unavailable"
        try:
            import lightning as L  # type: ignore
            li_ver = getattr(L, "__version__", "unknown")
        except Exception:
            li_ver = "unavailable"
        logging.info(
            "Lib versions -> torch=%s lightning=%s pytorch-forecasting=%s",
            getattr(torch, "__version__", "unknown"), li_ver, pf_ver,
        )

        logging.info("Predicting with TFT model (forecast horizon only, original-scale required)...")
        # Require original-scale predictions to avoid hidden scale mismatches.
        # We also request return_x to maximize availability of denormalized predictions in some PF versions.
        returns = model.predict(test_loader, return_index=True, mode="prediction", return_x=True)

        preds_pred = getattr(returns, "prediction", None)
        preds_tensor = None
        if preds_pred is not None:
            # Original-scale provided by PF
            if isinstance(preds_pred, list):
                if len(preds_pred) == 0:
                    raise RuntimeError("Prediction.prediction list is empty.")
                if not all(torch.is_tensor(o) for o in preds_pred):
                    raise RuntimeError("All elements in Prediction.prediction list must be tensors.")
                ndims = set(int(o.ndim) for o in preds_pred)
                if ndims == {3}:
                    preds_tensor = torch.cat(preds_pred, dim=0)  # list of batches
                elif ndims == {2}:
                    # list of horizon steps -> stack along time dim
                    if not all(preds_pred[0].shape == o.shape for o in preds_pred):
                        raise RuntimeError("Inconsistent shapes across Prediction.prediction step tensors")
                    preds_tensor = torch.stack(preds_pred, dim=1)  # (N, T, C)
                else:
                    raise RuntimeError(f"Unsupported list element dimensions for Prediction.prediction: {ndims}")
            elif torch.is_tensor(preds_pred):
                preds_tensor = preds_pred
            else:
                try:
                    preds_tensor = torch.as_tensor(preds_pred)
                except Exception:
                    raise RuntimeError(f"Unsupported Prediction.prediction type: {type(preds_pred)}")
            logging.info("TFT predict: using 'prediction' with shape %s", tuple(preds_tensor.shape))
        else:
            # PF 1.4.x path: require explicit inverse-scaling using target_scale in returns.x
            outputs = getattr(returns, "output", None)
            x_payload = getattr(returns, "x", None)
            if outputs is None:
                raise RuntimeError("Prediction.prediction missing and Prediction.output is None.")
            if x_payload is None:
                raise RuntimeError("Prediction.prediction missing and returns.x is None; cannot inverse-scale.")

            # Normalize outputs into a 3D tensor (n_samples, pred_len, out_size)
            if isinstance(outputs, list):
                if len(outputs) == 0:
                    raise RuntimeError("Prediction.output list is empty.")
                if not all(torch.is_tensor(o) for o in outputs):
                    conv = []
                    for o in outputs:
                        if torch.is_tensor(o):
                            conv.append(o)
                        else:
                            import numpy as _np
                            if isinstance(o, _np.ndarray):
                                conv.append(torch.from_numpy(o))
                            else:
                                raise RuntimeError("Prediction.output contained non-tensor, non-numpy element.")
                    outputs = conv
                ndims = set(int(o.ndim) for o in outputs)
                if ndims == {3}:
                    out_tensor = torch.cat(outputs, dim=0)  # list of batches
                elif ndims == {2}:
                    if len(outputs) == 1:
                        out_tensor = outputs[0].unsqueeze(1)
                    else:
                        if not all(outputs[0].shape == o.shape for o in outputs):
                            raise RuntimeError("Inconsistent shapes across Prediction.output step tensors")
                        out_tensor = torch.stack(outputs, dim=1)  # (N, T, C)
                else:
                    raise RuntimeError(f"Unsupported list element dimensions for Prediction.output: {ndims}")
            elif torch.is_tensor(outputs):
                out_tensor = outputs
            else:
                try:
                    out_tensor = torch.as_tensor(outputs)
                except Exception:
                    raise RuntimeError(f"Unsupported Prediction.output type: {type(outputs)}")
            if out_tensor.ndim == 2:
                out_tensor = out_tensor.unsqueeze(1)  # (N,1,C)
            if out_tensor.ndim != 3:
                raise RuntimeError(f"Expected 3D outputs for inverse-scaling, got shape {tuple(out_tensor.shape)}")

            # Establish shapes and detect which axis is targets vs horizon
            n_samples, d1, d2 = out_tensor.shape
            n_targets_ref = len(targets)
            # Prefer trained targets if present
            if trained_targets is not None:
                n_targets_ref = len(trained_targets)

            if d2 == n_targets_ref:
                # (N, pred_len, n_targets) -> transpose to (N, n_targets, pred_len)
                out_tensor = out_tensor.transpose(1, 2)
                n_targets, pred_len = d2, d1
            elif d1 == n_targets_ref:
                # (N, n_targets, pred_len)
                n_targets, pred_len = d1, d2
            else:
                raise RuntimeError(
                    f"Cannot identify target axis from outputs shape {tuple(out_tensor.shape)}; "
                    f"expected one dim to equal n_targets={n_targets_ref}"
                )
            out_size = n_targets  # alias for checks below

            # Strict validation: ensure model out_size matches trained targets if known
            if trained_targets is not None and len(trained_targets) != out_size:
                raise RuntimeError(
                    f"Model output channels ({out_size}) != trained targets ({len(trained_targets)}). "
                    f"Trained targets: {trained_targets} | Template targets: {targets}"
                )

            # Gather target_scale per sample from returns.x
            logging.info(
                "TFT predict: raw normalized outputs shape before inverse-scaling: %s",
                tuple(out_tensor.shape),
            )
            def _flatten_x_payload(x):
                if isinstance(x, dict):
                    return [x]
                if isinstance(x, list):
                    res = []
                    for item in x:
                        res.extend(_flatten_x_payload(item))
                    return res
                return []
            x_list = _flatten_x_payload(x_payload)
            if not x_list:
                raise RuntimeError(f"Unsupported or empty returns.x payload of type: {type(x_payload)}")

            scales = []
            for xb in x_list:
                if not isinstance(xb, dict):
                    raise RuntimeError("returns.x must be dicts containing 'target_scale'")
                ts = xb.get("target_scale")
                if ts is None:
                    raise RuntimeError("returns.x missing 'target_scale'; cannot inverse-scale outputs.")
                import numpy as _np
                if torch.is_tensor(ts):
                    ts_t = ts.detach().cpu()
                elif isinstance(ts, list):
                    elems = []
                    for u in ts:
                        if torch.is_tensor(u):
                            elems.append(u.detach().cpu())
                        elif isinstance(u, _np.ndarray):
                            elems.append(torch.from_numpy(u))
                        else:
                            elems.append(torch.as_tensor(u))
                    # If we got [center, scale] with shape (...), stack on last dim -> (..., 2)
                    try:
                        ts_t = torch.stack(elems, dim=-1)
                    except Exception as e:
                        raise RuntimeError(f"Failed to stack target_scale list elements: {e}")
                elif isinstance(ts, _np.ndarray):
                    ts_t = torch.from_numpy(ts)
                else:
                    ts_t = torch.as_tensor(ts)
                logging.info("TFT predict: batch target_scale raw shape: %s", tuple(ts_t.shape))
                scales.append(ts_t)
            try:
                target_scale = torch.cat(scales, dim=0)
            except Exception as e:
                raise RuntimeError(f"Failed to concatenate target_scale from returns.x: {e}")
            logging.info("TFT predict: concatenated target_scale raw shape: %s", tuple(target_scale.shape))

            # Normalize target_scale shape to (n_samples, n_targets, 2)
            if target_scale.ndim == 1:
                # Interpret as (2,) repeated per-sample not supported
                raise RuntimeError(f"Unexpected 1D target_scale: {tuple(target_scale.shape)}")
            if target_scale.ndim == 2:
                # (n_samples, 2) -> repeat per target channel
                target_scale = target_scale.unsqueeze(1).repeat(1, out_size, 1)
            elif target_scale.ndim == 3:
                if target_scale.size(-1) != 2:
                    # Possible transposed shape (n_samples, 2, out_size)
                    if target_scale.size(1) == 2:
                        target_scale = target_scale.transpose(1, 2)
                    else:
                        raise RuntimeError(f"Unexpected target_scale last dim != 2: {tuple(target_scale.shape)}")
                # Ensure middle dim matches out_size or can be broadcast
                if target_scale.size(1) not in (out_size, 1):
                    raise RuntimeError(
                        f"Unexpected target_scale middle dim {target_scale.size(1)} for out_size {out_size}"
                    )
                if target_scale.size(1) == 1:
                    target_scale = target_scale.repeat(1, out_size, 1)
            else:
                raise RuntimeError(f"Unexpected target_scale shape: {tuple(target_scale.shape)}")

            if target_scale.size(0) != n_samples:
                raise RuntimeError(
                    f"target_scale samples ({target_scale.size(0)}) != outputs samples ({n_samples}); cannot align."
                )
            # out_tensor currently (N, n_targets, pred_len). Broadcast scale on the last axis
            center = target_scale[..., 0].unsqueeze(-1).repeat(1, 1, pred_len)
            scale = target_scale[..., 1].unsqueeze(-1).repeat(1, 1, pred_len)
            center = center.to(out_tensor.device, dtype=out_tensor.dtype)
            scale = scale.to(out_tensor.device, dtype=out_tensor.dtype)
            preds_tensor = out_tensor * scale + center  # (N, n_targets, pred_len)
            # Downstream expects (N, pred_len, n_targets)
            preds_tensor = preds_tensor.transpose(1, 2)
            logging.info(
                "TFT predict: inverse-scaled outputs using returns.x['target_scale'] -> outputs %s, scale %s",
                tuple(out_tensor.shape), tuple(target_scale.shape)
            )

        index_attr = getattr(returns, 'index', None)
        if isinstance(index_attr, list):
            dfs = [d for d in index_attr if isinstance(d, pd.DataFrame) and not d.empty]
            if not dfs:
                raise RuntimeError("Prediction.index list is empty or has no valid DataFrames.")
            index_df = pd.concat(dfs, ignore_index=True)
        elif isinstance(index_attr, pd.DataFrame):
            if index_attr.empty:
                raise RuntimeError("Prediction.index DataFrame is empty.")
            index_df = index_attr.copy()
        else:
            raise RuntimeError(f"Unsupported Prediction.index type: {type(index_attr)}")

        # Normalize index DataFrame
        if time_idx_name not in index_df.columns:
            raise KeyError(f"Time index column '{time_idx_name}' not found in prediction index DataFrame")
        index_df = index_df.reset_index(drop=True)

        # Handle multi-step horizon predictions
        if torch.is_tensor(preds_tensor) and preds_tensor.ndim == 3:
            n_samples, pred_len, out_size = preds_tensor.shape
            if len(index_df) == n_samples and pred_len > 1:
                logging.info("Expanding index_df for multi-step horizon: samples=%d, pred_len=%d", n_samples, pred_len)
                # Build expanded index
                expanded_rows = []
                for i in range(n_samples):
                    base_row = index_df.iloc[i]
                    base_time = base_row[time_idx_name]
                    for h in range(pred_len):
                        new_row = base_row.copy()
                        new_row[time_idx_name] = base_time + h
                        expanded_rows.append(new_row)
                index_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
                preds_flat = preds_tensor.detach().cpu().numpy().reshape(n_samples * pred_len, out_size)
            else:
                preds_flat = preds_tensor.detach().cpu().numpy().reshape(-1, preds_tensor.shape[-1])
        else:
            preds_flat = preds_tensor.detach().cpu().numpy()
            if preds_flat.ndim == 1:
                preds_flat = preds_flat.reshape(-1, 1)

        # Merge with test data for target values and additional columns needed for plotting
        key_cols = group_id_fields + [time_idx_name]
        # Include Year column for plotting if it exists
        additional_cols = ['Year'] if 'Year' in test_data.columns else []
        ref_cols = [c for c in key_cols + targets + additional_cols if c in test_data.columns]
        horizon_df = index_df[key_cols].merge(
            test_data[ref_cols].drop_duplicates(key_cols),
            on=key_cols,
            how='left'
        )

        # Check alignment and extract target values
        if preds_flat.shape[0] != len(horizon_df):
            # This indicates a misalignment between dataset template indexing and prediction output.
            # Previously we truncated which masked the underlying issue and produced distorted metrics
            # (e.g., dramatically negative R^2). Raising an error forces investigation (likely due to
            # randomization differences or schema drift).
            raise RuntimeError(
                f"Prediction rows ({preds_flat.shape[0]}) != horizon_df rows ({len(horizon_df)}). "
                "This suggests index misalignment. Ensure stop_randomization=True for predict datasets "
                "and that test_data schema matches the saved template."
            )

        y_true = horizon_df[targets].values
        y_pred = preds_flat[:, : len(targets)] if preds_flat.shape[1] >= len(targets) else preds_flat

        # Valid mask: both sides finite & non-NaN
        valid_mask = (~np.isnan(y_true).any(axis=1)) & (~np.isnan(y_pred).any(axis=1))
        if valid_mask.any():
            save_metrics(run_id, y_true[valid_mask], y_pred[valid_mask])
        else:
            logging.warning("No valid rows for metric computation (all rows have NaNs).")

        # Store horizon info for plotting consistency
        session_state['horizon_df'] = horizon_df
        session_state['horizon_y_true'] = y_true

        return y_pred


# Maintain backward compatibility
build_datasets = build_datasets

__all__ = [
    "build_datasets",
    "hyperparameter_search_tft", 
    "train_final_tft",
    "predict_tft",
]