import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch

from src.utils.utils import load_session_state, setup_console_logging
from src.trainers.tft_model import load_tft_checkpoint
from src.trainers.tft_dataset import from_train_template, load_dataset_template


def as_tensor(x):
    if torch.is_tensor(x):
        return x
    try:
        return torch.as_tensor(x)
    except Exception:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return torch.from_numpy(x)
        raise


def stack_list_to_tensor(seq):
    if len(seq) == 0:
        raise RuntimeError("Empty list to stack")
    if all(torch.is_tensor(o) and o.ndim == 3 for o in seq):
        return torch.cat(seq, dim=0)
    if all(torch.is_tensor(o) and o.ndim == 2 for o in seq):
        if not all(seq[0].shape == o.shape for o in seq):
            raise RuntimeError("Inconsistent shapes across step tensors")
        return torch.stack(seq, dim=1)
    raise RuntimeError(f"Unsupported list element dimensions: {set(int(o.ndim) for o in seq if torch.is_tensor(o))}")


def inverse_scale(returns, targets):
    preds_pred = getattr(returns, "prediction", None)
    if preds_pred is not None:
        if isinstance(preds_pred, list):
            preds = stack_list_to_tensor(preds_pred)
        else:
            preds = as_tensor(preds_pred)
        if preds.ndim == 2:
            preds = preds.unsqueeze(1)
        if preds.ndim != 3:
            raise RuntimeError(f"Unexpected prediction shape {tuple(preds.shape)}")
        return preds

    outputs = getattr(returns, "output", None)
    x_payload = getattr(returns, "x", None)
    if outputs is None or x_payload is None:
        raise RuntimeError("Missing outputs or x for inverse-scaling")
    if isinstance(outputs, list):
        out_tensor = stack_list_to_tensor([as_tensor(o) for o in outputs])
    else:
        out_tensor = as_tensor(outputs)
    if out_tensor.ndim == 2:
        out_tensor = out_tensor.unsqueeze(1)
    if out_tensor.ndim != 3:
        raise RuntimeError(f"Expected 3D outputs, got {tuple(out_tensor.shape)}")

    # Detect axes: want (N, n_targets, pred_len)
    n_samples, d1, d2 = out_tensor.shape
    n_targets_ref = len(targets)
    if d2 == n_targets_ref:
        out_tensor = out_tensor.transpose(1, 2)
        n_targets, pred_len = d2, d1
    elif d1 == n_targets_ref:
        n_targets, pred_len = d1, d2
    else:
        raise RuntimeError(
            f"Cannot identify target axis from outputs shape {tuple(out_tensor.shape)}; expected one dim == {n_targets_ref}"
        )

    def _flatten_x(x):
        if isinstance(x, dict):
            return [x]
        if isinstance(x, list):
            res = []
            for it in x:
                res.extend(_flatten_x(it))
            return res
        return []

    x_list = _flatten_x(x_payload)
    if not x_list:
        raise RuntimeError("Empty returns.x payload")
    scales = []
    for xb in x_list:
        ts = xb.get("target_scale")
        if ts is None:
            raise RuntimeError("returns.x missing target_scale")
        if isinstance(ts, list):
            ts = torch.stack([as_tensor(u) for u in ts], dim=-1)
        else:
            ts = as_tensor(ts)
        scales.append(ts.detach().cpu())
    target_scale = torch.cat(scales, dim=0)

    # Normalize to (N, n_targets, 2)
    if target_scale.ndim == 2:
        target_scale = target_scale.unsqueeze(1).repeat(1, n_targets, 1)
    elif target_scale.ndim == 3:
        if target_scale.size(-1) != 2:
            if target_scale.size(1) == 2:
                target_scale = target_scale.transpose(1, 2)
            else:
                raise RuntimeError(f"Unexpected target_scale shape {tuple(target_scale.shape)}")
        if target_scale.size(1) not in (n_targets, 1):
            raise RuntimeError(
                f"target_scale middle dim {target_scale.size(1)} incompatible with n_targets {n_targets}"
            )
        if target_scale.size(1) == 1:
            target_scale = target_scale.repeat(1, n_targets, 1)
    else:
        raise RuntimeError(f"Unexpected target_scale shape {tuple(target_scale.shape)}")

    if target_scale.size(0) != n_samples:
        raise RuntimeError("target_scale sample count mismatch")

    center = target_scale[..., 0].unsqueeze(-1).repeat(1, 1, pred_len)
    scale = target_scale[..., 1].unsqueeze(-1).repeat(1, 1, pred_len)
    center = center.to(out_tensor.device, dtype=out_tensor.dtype)
    scale = scale.to(out_tensor.device, dtype=out_tensor.dtype)

    preds = out_tensor * scale + center
    preds = preds.transpose(1, 2)  # (N, pred_len, n_targets)
    return preds


def compute_alignment(y_true, y_pred, targets):
    # Flatten horizon to rows
    Yt = y_true.reshape(-1, y_true.shape[-1])
    Yp = y_pred.reshape(-1, y_pred.shape[-1])
    valid = (~np.isnan(Yt).any(axis=1)) & (~np.isnan(Yp).any(axis=1))
    Yt, Yp = Yt[valid], Yp[valid]
    if Yt.size == 0:
        return None
    # Correlation matrix (C x C)
    C = Yt.shape[1]
    corr = np.zeros((C, C), dtype=float)
    for i in range(C):
        for j in range(C):
            a, b = Yt[:, i], Yp[:, j]
            if np.std(a) == 0 or np.std(b) == 0:
                corr[i, j] = 0.0
            else:
                corr[i, j] = np.corrcoef(a, b)[0, 1]
    # Hungarian for max absolute correlation
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
    except Exception:
        # Fallback greedy
        row_ind, col_ind = [], []
        used = set()
        for i in range(C):
            j = int(np.argmax(np.abs(corr[i])))
            while j in used:
                corr[i, j] = -np.inf
                j = int(np.argmax(np.abs(corr[i])))
            used.add(j)
            row_ind.append(i)
            col_ind.append(j)
        row_ind, col_ind = np.array(row_ind), np.array(col_ind)

    mapping = {targets[i]: targets[j] for i, j in zip(row_ind, col_ind)}
    diag = corr[row_ind, col_ind]
    return {
        "corr_matrix": corr,
        "assignment": list(zip([targets[i] for i in row_ind], [targets[j] for j in col_ind], diag)),
        "is_identity": all(i == j for i, j in zip(row_ind, col_ind)),
        "avg_abs_corr": float(np.mean(np.abs(diag))),
    }


def main():
    ap = argparse.ArgumentParser(description="Diagnose TFT target alignment without retraining")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--max_batches", type=int, default=4)
    args = ap.parse_args()

    setup_console_logging()
    ss = load_session_state(args.run_id)
    if not ss:
        logging.warning("No session_state found; proceeding with minimal context")
    test_data = ss.get("test_data")
    targets = ss.get("targets")
    if targets is None:
        logging.error("session_state missing targets")
        return 2

    model = load_tft_checkpoint(args.run_id)
    template = load_dataset_template(args.run_id)
    test_ds = from_train_template(template, test_data if test_data is not None else pd.DataFrame(), mode="predict")
    loader = test_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

    logging.info("Running a few batches to collect predictions...")
    batches = []
    with torch.inference_mode():
        for i, _ in enumerate(loader):
            batches.append(i)
            if len(batches) >= args.max_batches:
                break
    # Use model.predict to get proper returns structure
    returns = model.predict(loader, return_index=True, mode="prediction", return_x=True)
    preds = inverse_scale(returns, targets)  # (N, T, C)

    # Build index
    index_attr = getattr(returns, 'index', None)
    if isinstance(index_attr, list):
        dfs = [d for d in index_attr if isinstance(d, pd.DataFrame) and not d.empty]
        index_df = pd.concat(dfs, ignore_index=True) if dfs else None
    elif isinstance(index_attr, pd.DataFrame):
        index_df = index_attr.copy()
    else:
        index_df = None

    time_idx = getattr(template, 'time_idx', 'Step')
    group_ids = list(getattr(template, 'group_ids', []))
    # Expand index if needed
    if index_df is not None and len(index_df) == preds.shape[0] and preds.shape[1] > 1:
        rows = []
        for i in range(len(index_df)):
            base = index_df.iloc[i]
            base_t = base[time_idx]
            for h in range(preds.shape[1]):
                r = base.copy()
                r[time_idx] = base_t + h
                rows.append(r)
        index_df = pd.DataFrame(rows)

    # Build y_true
    if test_data is None:
        logging.error("No test_data found in session_state to compute y_true")
        return 2
    key_cols = group_ids + [time_idx]
    ref_cols = key_cols + [c for c in targets if c in test_data.columns]
    horizon_df = index_df[key_cols].merge(
        test_data[ref_cols].drop_duplicates(key_cols), on=key_cols, how='left'
    )
    y_true = horizon_df[targets].values
    y_pred = preds.detach().cpu().numpy().reshape(-1, preds.shape[-1])
    y_true = y_true.reshape(-1, y_true.shape[-1])
    valid = (~np.isnan(y_true).any(axis=1)) & (~np.isnan(y_pred).any(axis=1))
    logging.info("Valid rows: %d / %d", int(valid.sum()), int(valid.size))

    diag = compute_alignment(y_true, y_pred, targets)
    if diag is None:
        logging.error("No valid rows for diagnosis.")
        return 1
    logging.info("Avg |corr| under best assignment: %.4f", diag["avg_abs_corr"]) 
    for a, b, c in diag["assignment"]:
        logging.info("map %-25s -> %-25s corr=%.3f", a, b, c)
    if not diag["is_identity"]:
        logging.warning("Target order appears mismatched (non-identity permutation).")
        return 3
    logging.info("Target order appears aligned (identity permutation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
