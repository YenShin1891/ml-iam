"""The TFT dataset must not need the warning filters the project used to carry.

Three third-party warnings were filtered instead of fixed.  Two were sklearn
feature-name mismatches: pytorch_forecasting fits covariate scalers on
one-column DataFrames but re-applies the target center/scale ones to bare
numpy slices per sample, 18 warning lines per sample.  The third announced
every group too short to fill a window, once per dataset build.

Fixing the scaling also exposed a real bug: pytorch_forecasting standardises
every real it is not told otherwise about, including the __observed mask
columns -- so MaskedRMSE received weights of roughly 0.7/-1.5 instead of 1/0
and the loss rewarded error on interpolated points.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from configs.data import INDEX_COLUMNS, OUTPUT_VARIABLES
from configs.models.tft import NamelessStandardScaler
from src.data.preprocess import observed_mask_columns
from src.trainers.tft_dataset import (
    DatasetTemplate,
    create_train_dataset,
    drop_underlength_groups,
    from_train_template,
)

REGIONS = ["World", "R10AFRICA", "R5ASIA", "KOR"]
FILTERED_MESSAGES = ("feature names", "Min encoder length")


def _frame(n_groups=8, short_first=False):
    rng = np.random.default_rng(0)
    rows = []
    for g in range(n_groups):
        # 15 steps: room for the 3-step encoder plus the full 12-step
        # horizon that predict mode demands of every group.
        n_steps = 3 if (short_first and g == 0) else 15
        for step in range(n_steps):
            row = {
                "Model": f"M{g % 4}", "Scenario": f"S{g}",
                "Region": REGIONS[g % 4], "Model_Family": f"FAM{g % 3}",
                "Step": step,
                "GDP": float(rng.normal(50, 10)),
                "Population": float(rng.normal(1000, 100)),
            }
            for i, target in enumerate(OUTPUT_VARIABLES):
                row[target] = float(rng.normal(i, 1))
                row[f"{target}__observed"] = float(rng.random() > 0.3)
            rows.append(row)
    return pd.DataFrame(rows)


def _session_state(data):
    return {
        "train_data": data,
        "features": ["GDP", "Population", "Region", "Model_Family"],
        "targets": list(OUTPUT_VARIABLES),
    }


def _formerly_filtered(caught):
    return [
        str(w.message) for w in caught
        if any(fragment in str(w.message) for fragment in FILTERED_MESSAGES)
    ]


def test_observed_masks_reach_the_model_as_zeros_and_ones():
    """Standardised, an unobserved element's loss weight goes negative."""
    dataset, _ = create_train_dataset(_session_state(_frame()))

    obs_positions = [
        dataset.reals.index(c) for c in observed_mask_columns(OUTPUT_VARIABLES)
    ]
    x, _ = dataset[0]
    values = np.asarray(x["x_cont"][:, obs_positions])

    assert set(np.unique(values)) <= {0.0, 1.0}


def test_dataset_build_and_samples_warn_about_nothing_we_used_to_filter():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataset, _ = create_train_dataset(_session_state(_frame()))
        for i in range(5):
            dataset[i]

    assert _formerly_filtered(caught) == []


def test_template_rebuild_warns_about_nothing_we_used_to_filter():
    """Eval and predict builds inherit the training dataset's scalers."""
    data = _frame()
    dataset, _ = create_train_dataset(_session_state(data))
    template = DatasetTemplate.from_dataset(dataset)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eval_dataset = from_train_template(template, _frame(short_first=True), mode="predict")
        for i in range(min(5, len(eval_dataset))):
            eval_dataset[i]

    assert _formerly_filtered(caught) == []


def test_short_groups_are_dropped_in_one_line_we_write(caplog):
    with caplog.at_level(logging.INFO), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_train_dataset(_session_state(_frame(short_first=True)))

    assert _formerly_filtered(caught) == []
    drops = [r.getMessage() for r in caplog.records if "cannot fill one" in r.getMessage()]
    assert drops == [
        "1 of 8 groups have fewer than 4 steps and cannot fill one "
        "encoder+prediction window; dropping them."
    ]


def test_drop_underlength_groups_drops_only_the_short_group():
    data = _frame(short_first=True)

    out = drop_underlength_groups(data, INDEX_COLUMNS, "Step", 4)

    assert set(out["Scenario"]) == {f"S{g}" for g in range(1, 8)}
    assert len(out) == len(data) - 3


def test_dropping_every_group_is_an_error_not_a_cryptic_assert():
    """pytorch_forecasting fails an empty index with an internal assert."""
    data = _frame()

    with pytest.raises(ValueError, match="Every group has fewer than 99 steps"):
        drop_underlength_groups(data, INDEX_COLUMNS, "Step", 99)


def test_drop_underlength_groups_leaves_long_groups_alone():
    data = _frame()

    assert drop_underlength_groups(data, INDEX_COLUMNS, "Step", 4) is data


def test_nameless_scaler_learns_no_feature_names():
    scaler = NamelessStandardScaler().fit(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))

    assert not hasattr(scaler, "feature_names_in_")
    assert scaler.transform(pd.DataFrame({"a": [2.0]}))[0, 0] == pytest.approx(0.0)
