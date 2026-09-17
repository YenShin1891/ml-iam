"""XGB SHAP values: rows with unobserved lags dropped, then normalised per target."""
import numpy as np
import pandas as pd


def test_each_target_sums_to_one_and_keeps_its_feature_ratios():
    from src.visualization.shap_xgb import normalise_shap_values

    rng = np.random.default_rng(0)
    raw = rng.normal(size=(50, 4, 2)) * np.array([1.0, 1000.0])
    out = normalise_shap_values(raw)

    assert np.allclose(np.abs(out).mean(axis=0).sum(axis=0), 1.0)
    # One positive scale per target: no feature column is rewritten.
    for i in range(raw.shape[2]):
        ratio = out[:, :, i] / raw[:, :, i]
        assert np.allclose(ratio, ratio[0, 0])
    assert not np.shares_memory(out, raw)


def test_all_zero_target_is_left_as_zeros():
    from src.visualization.shap_xgb import normalise_shap_values

    assert not normalise_shap_values(np.zeros((5, 3, 1))).any()


def test_rankings_list_cross_target_lags_under_their_own_name(tmp_path, monkeypatch):
    import src.visualization.shap_xgb as shap_xgb

    monkeypatch.setattr(shap_xgb, "get_run_root", lambda _run_id: str(tmp_path))
    targets = ["A", "B"]
    features = ["prev_A", "prev_B", "Price|Carbon"]
    values = np.zeros((4, 3, 2))
    values[:, :, 0] = [0.2, 0.7, 0.1]  # target A leans on B's lag
    values[:, :, 1] = [0.1, 0.3, 0.6]
    shap_xgb.write_shap_rankings("xgb_00", values, targets, features)

    csv_dir = tmp_path / "plots" / "csv"
    ranking = pd.read_csv(csv_dir / "shap1_A.csv")
    assert list(ranking["Feature"]) == ["prev_B", "prev_A", "Price|Carbon"]
    assert np.allclose(ranking["Importance"], [0.7, 0.2, 0.1])
    assert not (csv_dir / "feature_renaming.json").exists()


def test_rows_with_an_unobserved_lag_are_dropped_and_other_gaps_kept():
    from src.visualization.shap_xgb import drop_unobserved_lag_rows

    frame = pd.DataFrame({
        "Scenario": ["a", "b", "c", "d"],
        "prev_A": [1.0, np.nan, 1.0, 1.0],
        "prev2_A": [1.0, 1.0, np.nan, 1.0],
        "Price|Carbon": [np.nan, 1.0, 1.0, 1.0],  # an exogenous gap is a real input
    })
    kept = drop_unobserved_lag_rows(frame, ["prev_A", "prev2_A", "Price|Carbon"])
    assert list(kept["Scenario"]) == ["a", "d"]


def test_frame_without_lag_features_is_returned_whole():
    from src.visualization.shap_xgb import drop_unobserved_lag_rows

    frame = pd.DataFrame({"Price|Carbon": [np.nan, 1.0]})
    assert len(drop_unobserved_lag_rows(frame, ["Price|Carbon"])) == 2
