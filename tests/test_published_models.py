"""Publishing trained runs: saved imputation medians, new-scenario tables, bundle install."""

import json
import tarfile

import numpy as np
import pandas as pd
import pytest

import src.data.preprocess as preprocess
from src.inference import new_inputs


# ── Imputation medians travel as numbers ──────────────────────────────────


def _split_frames():
    rng = np.random.default_rng(0)
    def frame(n, scales):
        return pd.DataFrame({
            "Region_Scale": rng.choice(scales, n),
            "Region": "R", "Model_Family": "F", "Year": 2020,
            "a": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)),
            "b": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)),
            "a_is_missing": 0.0,
        })
    return frame(200, ["World", "R5", "ISO3"]), frame(40, ["World", "R5"]), frame(40, ["ISO3"])


@pytest.mark.parametrize("scale_aware", [True, False])
def test_saved_medians_impute_exactly_as_the_training_pass_did(monkeypatch, scale_aware):
    import configs.data as data_config

    monkeypatch.setattr(data_config, "SCALE_AWARE_IMPUTATION", scale_aware)
    features = ["Region", "Model_Family", "a", "b", "a_is_missing"]
    train, val, test = _split_frames()
    medians = preprocess.compute_train_medians(train, features)
    medians = json.loads(json.dumps(medians))  # as read back from the run
    unseen = test.copy()

    _, _, imputed = preprocess.impute_with_train_medians(train.copy(), val.copy(), test, features)
    preprocess.apply_medians(unseen, medians, features)

    pd.testing.assert_frame_equal(unseen, imputed)
    assert unseen[["a", "b"]].notna().all().all()
    assert (medians["by_scale"] is not None) == scale_aware


def test_a_scale_the_medians_never_saw_falls_back_to_the_global_median():
    features = ["a"]
    train = pd.DataFrame({"Region_Scale": ["World"] * 3, "a": [1.0, 2.0, 3.0]})
    medians = preprocess.compute_train_medians(train, features)
    new = pd.DataFrame({"Region_Scale": ["R10"], "a": [np.nan]})

    preprocess.apply_medians(new, medians, features)

    assert new.loc[0, "a"] == 2.0


# ── Scenario tables ───────────────────────────────────────────────────────


def _table():
    return pd.DataFrame({
        "Model": ["MESSAGE-X 1.0"] * 2, "Scenario": ["s"] * 2, "Region": ["R5ASIA", "World"],
        "Variable": ["GDP|PPP", "GDP|PPP"], "2020": [1.0, 2.0], "2025": ["3", None],
    })


def test_a_table_gains_the_columns_that_can_be_derived():
    table = new_inputs.complete_scenario_table(_table())

    assert table["Model_Family"].tolist() == ["MESSAGE", "MESSAGE"]
    assert table["Region_Scale"].tolist() == ["R5", "World"]
    assert table["Scenario_Category"].eq(new_inputs.PLACEHOLDER_CATEGORY).all()
    assert table["2025"].tolist()[0] == 3.0 and np.isnan(table["2025"].tolist()[1])


def test_a_table_without_years_or_identifiers_is_refused_with_the_reason():
    with pytest.raises(new_inputs.InputError, match="Variable"):
        new_inputs.complete_scenario_table(_table().drop(columns="Variable"))
    with pytest.raises(new_inputs.InputError, match="year columns"):
        new_inputs.complete_scenario_table(_table().drop(columns=["2020", "2025"]))
    with pytest.raises(new_inputs.InputError, match="duplicated"):
        new_inputs.complete_scenario_table(pd.concat([_table(), _table()]))


def test_input_variables_leave_out_what_the_pipeline_derives():
    features = ["Region", "Model_Family", "GDP|PPP", "GDP|PPP_is_missing", "prev_Emissions|CO2",
                "prev2_Emissions|CO2", "DeltaYears"]

    assert new_inputs.input_variables(features) == ["GDP|PPP"]


def test_an_unseen_region_is_named_rather_than_embedded_at_random():
    long = pd.DataFrame({"Region": ["ATLANTIS"], "Model_Family": ["AIM"]})

    with pytest.raises(new_inputs.InputError, match="ATLANTIS"):
        new_inputs._check_vocabulary(long, {"Region": ["World"], "Model_Family": ["AIM"]})


def test_absent_variables_become_unreported_columns_and_empty_years_are_not_timesteps():
    notes = []
    table = new_inputs.complete_scenario_table(_table().assign(**{"2030": [None, None]}))

    long = new_inputs._long_frame(table, ["GDP|PPP", "Population"], ["Emissions|CO2"], notes)

    assert long["Population"].isna().all() and long["Emissions|CO2"].isna().all()
    assert sorted(long["Year"].unique()) == [2020, 2025]
    assert any("Population" in note for note in notes)


def test_tft_group_labels_are_swapped_for_known_ones_one_pair_each():
    class Encoder:
        def __init__(self, labels):
            self.classes_ = {label: i for i, label in enumerate(labels)}

    class Template:
        categorical_encoders = {"__group_id__Model": Encoder(["m0", "m1"]),
                                "__group_id__Scenario": Encoder(["s0", "s1", "s2"])}

    frame = pd.DataFrame({"Model": ["new"] * 4, "Scenario": ["a", "a", "b", "c"], "Region": ["World"] * 4})

    aliased, pairs = new_inputs._alias_group_ids(frame, Template())

    assert set(aliased["Model"]) <= {"m0", "m1"} and set(aliased["Scenario"]) <= {"s0", "s1", "s2"}
    assert len(pairs) == 3 and not pairs[["_alias_model", "_alias_scenario"]].duplicated().any()
    assert aliased.groupby(["Model", "Scenario"]).ngroups == 3


# ── Bundles ───────────────────────────────────────────────────────────────


def _bundle(tmp_path, run_id="xgb_01", extra=None):
    from scripts.fetch_models import _sha256

    root = tmp_path / "build" / run_id
    (root / "artifacts").mkdir(parents=True)
    (root / "artifacts" / "features.json").write_text("{}")
    manifest = {"artifacts/features.json": _sha256(root / "artifacts" / "features.json")}
    (root / "manifest.json").write_text(json.dumps(manifest))
    archive = tmp_path / f"{run_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root, arcname=run_id)
        for name, source in (extra or {}).items():
            tar.add(source, arcname=name)
    return archive, _sha256(archive)


def test_a_bundle_is_installed_under_its_model_directory_and_not_twice(tmp_path):
    from scripts.fetch_models import install

    archive, digest = _bundle(tmp_path)
    results = tmp_path / "results"

    target = install("xgb_01", archive, digest, results)

    assert target == results / "xgb" / "xgb_01" and (target / "artifacts" / "features.json").exists()
    with pytest.raises(FileExistsError):
        install("xgb_01", archive, digest, results)


def test_a_bundle_with_the_wrong_checksum_or_a_stray_member_is_refused(tmp_path):
    from scripts.fetch_models import install

    archive, digest = _bundle(tmp_path)
    with pytest.raises(RuntimeError, match="SHA-256"):
        install("xgb_01", archive, "0" * 64, tmp_path / "results")

    stray = tmp_path / "stray.txt"
    stray.write_text("x")
    (tmp_path / "second").mkdir()
    archive, digest = _bundle(tmp_path / "second", extra={"elsewhere/stray.txt": stray})
    with pytest.raises(RuntimeError, match="outside"):
        install("xgb_01", archive, digest, tmp_path / "results2")


def test_the_export_scan_finds_local_paths_but_not_torch_archive_names(tmp_path):
    from scripts.export_run_bundle import _scan_for_local_paths

    (tmp_path / "ok.ckpt").write_bytes(b"PK archive/data/0FB archive/data/1FB")
    assert _scan_for_local_paths(tmp_path) == []

    (tmp_path / "leaky.json").write_text('{"dirpath": "/mnt/storage/results/tft/tft_01/final"}')
    assert [name for name, _ in _scan_for_local_paths(tmp_path)] == ["leaky.json"]
