"""Local UI preview with explicitly synthetic data; never loads a trained model.

Run: python -m streamlit run scripts/dashboard_whatif_preview.py
The real view's controls, run actions, exports and plots are reused below.
"""

from types import SimpleNamespace
from functools import partial
import tempfile
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from configs.dashboard import WHATIF_COMBINATION_DEFAULTS, WHATIF_KEY_LEVERS, WHATIF_COMPARISON_OUTPUTS, WHATIF_DISTRIBUTION_YEAR
from configs.data import OUTPUT_VARIABLES
from scripts import dashboard_whatif as view
from src.inference import engines
from src.inference.whatif import (
    WHATIF_CATEGORY_GROUPS, PreparedRun, BaselineCandidate, apply_levers, assemble_result, build_lever_specs,
    ar6_distribution_samples, baseline_rows, candidate_baselines, high_low_combinations, lever_bands, resolve_anchor_years,
)
from src.visualization.whatif import plot_whatif_comparison, save_whatif_outputs

# Source-input names from the local tft_86 artifacts/features.json (42 inputs).
# This representative schema is not a verified tft_94 schema. Lagged targets,
# categories and missingness indicators are not user-adjustable source inputs.
PREVIEW_FEATURES = [
    "Capital Cost|Electricity|Biomass|w/ CCS",
    "Capital Cost|Electricity|Biomass|w/o CCS",
    "Capital Cost|Electricity|Coal|w/ CCS",
    "Capital Cost|Electricity|Coal|w/o CCS",
    "Capital Cost|Electricity|Gas|w/ CCS",
    "Capital Cost|Electricity|Gas|w/o CCS",
    "Capital Cost|Electricity|Geothermal",
    "Capital Cost|Electricity|Hydro",
    "Capital Cost|Electricity|Nuclear",
    "Capital Cost|Electricity|Solar|CSP",
    "Capital Cost|Electricity|Solar|PV",
    "Capital Cost|Electricity|Wind|Offshore",
    "Capital Cost|Electricity|Wind|Onshore",
    "Capital Cost|Hydrogen|Electricity",
    "Efficiency|Electricity|Biomass|w/ CCS",
    "Efficiency|Electricity|Biomass|w/o CCS",
    "Efficiency|Electricity|Coal|w/ CCS",
    "Efficiency|Electricity|Coal|w/o CCS",
    "Efficiency|Electricity|Gas|w/ CCS",
    "Efficiency|Electricity|Gas|w/o CCS",
    "GDP|MER", "GDP|PPP",
    "Lifetime|Electricity|Geothermal",
    "Lifetime|Electricity|Hydro",
    "Lifetime|Electricity|Nuclear",
    "Lifetime|Electricity|Solar|PV",
    "OM Cost|Fixed|Electricity|Biomass|w/ CCS",
    "OM Cost|Fixed|Electricity|Biomass|w/o CCS",
    "OM Cost|Fixed|Electricity|Coal|w/ CCS",
    "OM Cost|Fixed|Electricity|Coal|w/o CCS",
    "OM Cost|Fixed|Electricity|Gas|w/ CCS",
    "OM Cost|Fixed|Electricity|Gas|w/o CCS",
    "OM Cost|Fixed|Electricity|Geothermal",
    "OM Cost|Fixed|Electricity|Hydro",
    "OM Cost|Fixed|Electricity|Nuclear",
    "OM Cost|Fixed|Electricity|Solar|CSP",
    "OM Cost|Fixed|Electricity|Solar|PV",
    "OM Cost|Fixed|Electricity|Wind|Offshore",
    "OM Cost|Fixed|Electricity|Wind|Onshore",
    "Population", "Price|Carbon", "Yield|Cereal",
]


def preview_data():
    years = np.arange(2005, 2101, 5)
    t = np.linspace(0, 1, len(years))
    rows = pd.DataFrame({
        "Year": years, "Step": np.arange(len(years)), "Model": "Preview IAM",
        "Scenario": "Illustrative baseline", "Region": "World", "Scenario_Category": "C3",
        "Model_Family": "Preview", "split": "test",
    })
    for i, feature in enumerate(PREVIEW_FEATURES):
        if feature.startswith("Efficiency|"):
            start, end = .30, .55
        elif feature.startswith("Lifetime|"):
            start, end = 25., 40.
        elif feature.startswith("OM Cost|"):
            start, end = 50. + i, 30. + i
        elif feature == "GDP|PPP":
            start, end = 95000., 330000.
        else:
            start, end = 2400., 1400.
        rows[feature] = start + (end - start) * t
        rows[f"{feature}_is_missing"] = 0.0
    for feature, start, end in zip(
        WHATIF_KEY_LEVERS, [6500, 60000, 20, 4, 1800, 1600, 5000, 2200, 1200],
        [10000, 220000, 250, 7, 450, 850, 4200, 1900, 1100],
    ):
        rows[feature] = start + (end - start) * t
        rows[f"{feature}_is_missing"] = 0.0
    paths = [180000*(1-.8*t), 110000*(1-.45*t), 160000*(1-.65*t),
             3000+125000*t**1.5, 4000+90000*t**1.3, 25000+22000*t,
             36000*(1-.88*t), 350*(1-.4*t), 12*(1-.25*t)]
    for target, values in zip(OUTPUT_VARIABLES, paths):
        rows[target] = values
        rows[f"{target}__observed"] = 1.0
    population = pd.concat([
        rows.assign(**{f: rows[f] * scale for f in PREVIEW_FEATURES}, Scenario=f"Preview {scale}")
        for scale in np.linspace(.55, 1.6, 12)
    ], ignore_index=True)
    anchors = resolve_anchor_years(years.tolist(), years[:3].tolist())
    specs = build_lever_specs(rows, lever_bands(population, PREVIEW_FEATURES), PREVIEW_FEATURES, 3, anchors)
    return rows, specs, anchors


def illustrative_prediction(frame, original):
    """Deterministic drawing data, not TFT inference or a scientific model."""
    change = np.zeros(len(frame))
    for feature, weight in zip(WHATIF_COMBINATION_DEFAULTS, [-.7, .5, .4, .65]):
        change += weight * (frame[feature].to_numpy() / original[feature].to_numpy() - 1)
    # Let every original slider exercise the preview flow. These arbitrary
    # drawing weights do not represent the model's learned input sensitivity.
    for feature in PREVIEW_FEATURES:
        if feature not in WHATIF_COMBINATION_DEFAULTS:
            change += .02 * (frame[feature].to_numpy() / original[feature].to_numpy() - 1)
    out = frame[["Year", "Step"]].iloc[3:].copy()
    for i, target in enumerate(OUTPUT_VARIABLES):
        sign = -1 if i in (3, 4) else 1
        out[f"{target}_pred"] = original[target].to_numpy()[3:] * 1.02 * np.exp(sign * change[3:])
    return out


def preview_population():
    """Three explicitly illustrative category groups plus reference sample clouds."""
    rows, _, _ = preview_data()
    frames = []
    for k, category in enumerate(("C3", "C5", "C7")):
        baseline = rows.copy()
        baseline["Scenario_Category"] = category
        baseline["Scenario"] = "Illustrative baseline" if k == 0 else f"Illustrative {category} baseline"
        t = np.linspace(0, 1, len(baseline))
        for i, target in enumerate(OUTPUT_VARIABLES):
            # Illustration only: progressively different category trajectories.
            if i in (3, 4):
                baseline[target] *= 1 - .22 * k * t
            else:
                baseline[target] += rows[target].iloc[0] * .5 * k * t
        frames.append(baseline)
        for j, scale in enumerate(np.linspace(.65, 1.4, 18)):
            sample = baseline.copy()
            sample["Scenario"] = f"Reference {category} {j:02d}"
            for feature in PREVIEW_FEATURES:
                sample[feature] *= scale
            for target in OUTPUT_VARIABLES:
                sample[target] *= 1 + (scale - 1) * t
            frames.append(sample)
    model_features = PREVIEW_FEATURES + [f"{f}_is_missing" for f in PREVIEW_FEATURES] + ["Model_Family", "Region"]
    return PreparedRun("preview", pd.concat(frames, ignore_index=True), model_features,
                       list(PREVIEW_FEATURES), list(OUTPUT_VARIABLES))


def preview_ensembles(prepared, groups=None, features=None, selected_labels=None):
    groups = list(WHATIF_CATEGORY_GROUPS) if groups is None else groups
    features = WHATIF_COMBINATION_DEFAULTS if features is None else features
    selected_labels = selected_labels or {}
    bands = lever_bands(prepared.frame, prepared.raw_features)
    ensembles = {}
    for group in groups:
        candidates = [c for category in WHATIF_CATEGORY_GROUPS[group] for c in candidate_baselines(prepared, "World", category=category)]
        candidate = next((c for c in candidates if c.label == selected_labels.get(group)), candidates[0])
        rows = baseline_rows(prepared, "World", candidate.model, candidate.scenario)
        years = rows.Year.tolist()
        specs = build_lever_specs(rows, bands, prepared.raw_features, 3, resolve_anchor_years(years, years[:3]))
        pred = illustrative_prediction(rows, rows)
        original = assemble_result("preview", "World", candidate, rows, rows, prepared.raw_features,
                                   prepared.targets, pred, pred, {}, (0, 0), 3)
        forecasts = {}
        for label, edits in high_low_combinations(specs, features):
            frame = illustrative_prediction(apply_levers(rows, edits, 3), rows)
            forecasts[label] = frame.set_index("Year").filter(like="_pred").rename(columns=lambda c: c[:-5])
        ensembles[group] = {"original": original, "predictions": forecasts}
    return ensembles


def main():
    st.set_page_config(page_title="ML-IAM revision preview", layout="wide")
    st.title("ML-IAM · Revision preview")
    st.warning("UI PREVIEW — all data and curves on this page are synthetic examples, not tft_94 predictions.")
    st.caption("Local preview only. Changes have not been pushed or deployed to mliam.dev.")
    st.caption("Input coverage: 42 source inputs from an older local run's schema; 41 controls with PPP hidden. Values are synthetic. The tft_94 schema has not been verified here.")
    with st.sidebar:
        st.header("Emulation Viewer")
        st.selectbox("View", ["What-if emulator"])
        st.text("Revision · local preview")
    st.selectbox("Region", ["World"])
    st.selectbox("Baseline scenario", ["Illustrative baseline (synthetic)"])
    rows, specs, anchor_years = preview_data()
    candidate = BaselineCandidate("Preview IAM", "Illustrative baseline", "test", len(rows), 2005, 2100, len(specs), 1)
    prepared = preview_population()
    engine = SimpleNamespace(encoder_length=3, min_steps=15)
    pred = illustrative_prediction(rows, rows)
    original = assemble_result("preview", "World", candidate, rows, rows, prepared.raw_features,
                               prepared.targets, pred, pred, {}, (0, 0), 3)
    view._state_default("whatif_gen", 0)
    view._state_default("whatif_edits", {})
    view._state_default("whatif_preset", None)
    view._state_default("whatif_result", None)
    view._state_default("whatif_saved", None)
    st.session_state.whatif_specs = {spec.feature: spec for spec in specs}
    view._render_controls(specs, anchor_years)
    view._render_gdp_values(rows, 3)
    view._render_overlay(specs)
    # Only this separate preview entry point replaces inference with drawing data.
    def source_for(frame):
        return baseline_rows(prepared, "World", frame.Model.iloc[0], frame.Scenario.iloc[0])

    def predict_baseline(_run, region, model, scenario):
        source = baseline_rows(prepared, region, model, scenario)
        return illustrative_prediction(source, source)

    if "preview_output_dir" not in st.session_state:
        st.session_state.preview_output_dir = tempfile.mkdtemp(prefix="mliam-preview-")
    with patch.object(view, "_baseline_prediction", side_effect=predict_baseline), \
         patch.object(view, "_prepared", return_value=prepared), \
         patch.object(view, "save_whatif_outputs", partial(save_whatif_outputs, output_dir=st.session_state.preview_output_dir)), \
         patch.object(view, "_ar6_band", return_value=None), \
         patch.object(view, "_bands", return_value=lever_bands(prepared.frame, prepared.raw_features)), \
         patch.object(engines, "check_vocabulary", return_value=[]), \
         patch.object(engines, "predict_windows", side_effect=lambda _engine, frame: illustrative_prediction(frame, source_for(frame))):
        view._render_run("preview", engine, prepared, "World", candidate, rows, 3, specs, float("nan"), None)
        view._render_combinations("preview", engine, prepared, "World", candidate, rows, 3, specs, float("nan"), None)
    if st.session_state.get("whatif_combinations") is None:
        groups = list(WHATIF_CATEGORY_GROUPS)
        features = WHATIF_COMBINATION_DEFAULTS
        outputs = [t for t in WHATIF_COMPARISON_OUTPUTS if t in prepared.targets]
        year = WHATIF_DISTRIBUTION_YEAR
        st.markdown("**Example result · category groups, time series and distributions**")
        ensembles = preview_ensembles(prepared, groups, features)
        reference = ar6_distribution_samples(prepared.frame, "World", prepared.targets, year)
        fig = plot_whatif_comparison(ensembles, reference, year=year, targets=outputs)
        fig.suptitle("SYNTHETIC PREVIEW — chart layout only, not tft_94 predictions")
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
