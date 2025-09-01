
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Any

import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer

from xgboost import XGBRegressor, XGBClassifier

def _merge_fit_params_with_fold_eval(fit_params: Optional[Dict[str, Any]], X_val, y_val) -> Dict[str, Any]:
    params = dict(fit_params) if fit_params else {}
    wants_es = any(k in params for k in ("early_stopping_rounds",))
    if wants_es:
        params.setdefault("eval_set", [(X_val, y_val)])
        params.setdefault("eval_metric", "rmse")
        params.setdefault("verbose", False)
    return params

def _evaluate_param_set(
    X, y, groups, estimator_class, base_estimator_kwargs: Dict[str, Any],
    params: Dict[str, Any], cv, scoring: Optional[str], random_state: int, fit_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    scorer = get_scorer(scoring) if scoring is not None else None
    fold_scores: List[float] = []
    for tr_idx, va_idx in cv.split(X, y, groups):
        Xtr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
        Xva = X.iloc[va_idx] if hasattr(X, "iloc") else X[va_idx]
        ytr = y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx]
        yva = y.iloc[va_idx] if hasattr(y, "iloc") else y[va_idx]

        # Ensure aligned indices for pandas
        if hasattr(Xtr, "reset_index"): Xtr = Xtr.reset_index(drop=True)
        if hasattr(Xva, "reset_index"): Xva = Xva.reset_index(drop=True)
        if hasattr(ytr, "reset_index"): ytr = ytr.reset_index(drop=True)
        if hasattr(yva, "reset_index"): yva = yva.reset_index(drop=True)

        est_kwargs = dict(base_estimator_kwargs)
        est_kwargs.update(params)
        est_kwargs.setdefault("random_state", random_state)
        model = estimator_class(**est_kwargs)

        per_fold_fit_params = _merge_fit_params_with_fold_eval(fit_params, Xva, yva)
        model.fit(Xtr, ytr, **per_fold_fit_params)

        if scorer is None:
            score = float(model.score(Xva, yva))
        else:
            score = float(scorer(model, Xva, yva))
        fold_scores.append(score)

    mean_score = float(np.mean(fold_scores))
    return {"params": params, "mean_score": mean_score, "fold_scores": fold_scores}

def dask_random_search_like_sklearn(
    X, y, *,
    param_distributions: Dict[str, Iterable[Any]],
    n_iter: int = 20,
    estimator: str = "regressor",
    base_estimator_kwargs: Optional[Dict[str, Any]] = None,
    cv=None,
    groups=None,
    scoring: Optional[str] = None,
    random_state: int = 42,
    fit_params: Optional[Dict[str, Any]] = None,
    client=None,
    refit: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate param_distributions with the SAME CV splitter and scoring semantics as scikit-learn.
    Folds are evaluated sequentially within each task to keep behavior close to sklearn.
    """
    if base_estimator_kwargs is None:
        base_estimator_kwargs = {}

    estimator_class = XGBClassifier if estimator == "classifier" else XGBRegressor
    if cv is None:
        cv = KFold(n_splits=5, shuffle=False)

    # Build a deterministic sampler identical to sklearn's ParameterSampler
    from sklearn.model_selection import ParameterSampler
    sampler = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))

    # Coarse-grained parallelism: one task per param set
    if client is not None:
        futures = [client.submit(
            _evaluate_param_set, X, y, groups, estimator_class, base_estimator_kwargs,
            params, cv, scoring, random_state, fit_params, pure=False
        ) for params in sampler]
        results = client.gather(futures)
    else:
        results = [
            _evaluate_param_set(X, y, groups, estimator_class, base_estimator_kwargs,
                                params, cv, scoring, random_state, fit_params)
            for params in sampler
        ]

    best = max(results, key=lambda r: r["mean_score"]) if results else None
    out = {
        "results": results,
        "best_params": best["params"] if best else None,
        "best_score": best["mean_score"] if best else None,
    }

    if refit and best is not None:
        est_kwargs = dict(base_estimator_kwargs)
        est_kwargs.update(best["params"])
        est_kwargs.setdefault("random_state", random_state)
        best_estimator = estimator_class(**est_kwargs)
        best_estimator.fit(X, y, **(fit_params or {}))
        out["best_estimator"] = best_estimator

    return out
