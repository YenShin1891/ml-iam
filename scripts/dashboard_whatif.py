"""The dashboard's what-if view: move a scenario's inputs and run the emulator live.

Imported by scripts/dashboard.py.  The trained model is loaded once per
process and kept on the CPU; the frames the run was fitted on are rebuilt
once too.  Torch is imported only inside the functions that run the model,
so the trajectories view works without it.
"""

import logging
import os
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from configs.dashboard import (
    WHATIF_ACCELERATOR,
    WHATIF_BASELINE_CATEGORY,
    WHATIF_DEFAULT_REGION,
    WHATIF_ENGINES,
    WHATIF_KEY_LEVERS,
    WHATIF_MIN_SAMPLE_SIZE,
    WHATIF_PRESETS,
    WHATIF_R2_THRESHOLD,
)
from src.inference.whatif import (
    R2_COLUMN,
    SAMPLE_COLUMN,
    LeverSpec,
    apply_levers,
    ar6_target_bands,
    assemble_result,
    band_coverage,
    baseline_rows,
    build_lever_specs,
    candidate_baselines,
    choose_default_baseline,
    eligible_regions,
    export_frame,
    lever_bands,
    load_prepared_run,
    multiplier_from_anchors,
    preset_edits,
    ramp_anchor_values,
    region_metrics_from_predictions,
    resolve_anchor_years,
    result_metadata,
)
from src.utils.run_store import RunStore
from src.visualization.whatif import plot_lever_overlay, plot_whatif_grid, save_whatif_outputs

VIEW_NAME = "What-if emulator"

# Keys that outlive a change of run: the view stays, and the visitor's
# region and baseline are re-validated against the new run's options.
# Streamlit forgets a widget's own key while the widget is not rendered (a
# detour through an XGB run, say), so the choices are also kept in plain
# keys that seed the widgets when they come back.
PRESERVED_KEYS = (
    "dashboard_view",
    "whatif_region",
    "whatif_region_choice",
    "whatif_baseline",
    "whatif_baseline_choice",
    "whatif_advanced",
    "whatif_overlay_mode",
    "whatif_show_band",
)

OVERLAY_MODES = {"Position within the AR6 range": "band", "Relative to the baseline": "ratio"}

CAVEAT = (
    "This emulator learned from AR6 scenarios and reproduces their behaviour. It does not "
    "guarantee accuracy for inputs outside the range those scenarios span. Only regions where "
    "the run reaches a pooled R² of at least {threshold:.2f} on held-out scenarios (scored on at "
    "least {min_samples:,} values) are offered, and every lever is bounded by the 5–95% range of "
    "the training scenarios in that region and year."
)


# ── Cached loads ──────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading the emulator…")
def _engine(run_id: str):
    from src.inference.tft_predict import load_engine

    return load_engine(run_id, map_location=WHATIF_ACCELERATOR)


@st.cache_resource(show_spinner="Rebuilding the run's scenarios…")
def _prepared(run_id: str):
    return load_prepared_run(RunStore(run_id))


@st.cache_data(show_spinner=False, ttl=300)
def _region_table(run_id: str) -> Optional[pd.DataFrame]:
    """The per-region metrics: the saved table, else computed from the loaded bundle."""
    store = RunStore(run_id)
    if store.has_metrics_by_region():
        return store.load_metrics_by_region()
    state = st.session_state
    if all(state.get(key) is not None for key in ("horizon_df", "horizon_y_true", "preds", "targets")):
        return region_metrics_from_predictions(
            run_id, state.horizon_df, state.horizon_y_true, state.preds, state.targets
        )
    return None


@st.cache_data(show_spinner=False)
def _candidates(run_id: str, region: str):
    return candidate_baselines(_prepared(run_id), region)


@st.cache_data(show_spinner=False)
def _bands(run_id: str, region: str) -> pd.DataFrame:
    prepared = _prepared(run_id)
    frame = prepared.frame
    rows = frame[(frame["Region"] == region) & (frame["split"] == "train")]
    return lever_bands(rows, prepared.raw_features)


@st.cache_data(show_spinner=False)
def _ar6_band(run_id: str, region: str) -> pd.DataFrame:
    prepared = _prepared(run_id)
    frame = prepared.frame
    return ar6_target_bands(frame[frame["Region"] == region], prepared.targets)


@st.cache_data(show_spinner="Emulating the unchanged inputs…")
def _baseline_prediction(run_id: str, region: str, model: str, scenario: str) -> pd.DataFrame:
    from src.inference.tft_predict import predict_windows

    rows = baseline_rows(_prepared(run_id), region, model, scenario)
    return predict_windows(_engine(run_id), rows)


# ── Session state ─────────────────────────────────────────────────────────


def _state_default(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _invalidate_result() -> None:
    st.session_state.whatif_result = None
    st.session_state.whatif_saved = None


def _bump_generation() -> None:
    """Re-create every lever widget on the next run, so it reads the new edits."""
    st.session_state.whatif_gen = st.session_state.get("whatif_gen", 0) + 1


def _widget_key(name: str) -> str:
    return f"whatif_w_{st.session_state.whatif_gen}_{name}"


def _specs() -> Dict[str, LeverSpec]:
    return st.session_state.whatif_specs


def _on_multiplier_change(feature: str, key: str) -> None:
    spec = _specs()[feature]
    multiplier = float(st.session_state[key])
    edits = st.session_state.whatif_edits
    if abs(multiplier - 1.0) < 1e-9:
        edits.pop(feature, None)
    else:
        edits[feature] = ramp_anchor_values(spec, multiplier)
    st.session_state.whatif_preset = None
    _invalidate_result()


def _on_anchor_change(feature: str, year: int, key: str) -> None:
    spec = _specs()[feature]
    edits = st.session_state.whatif_edits
    anchors = edits.setdefault(feature, {y: b.base for y, b in spec.anchors.items()})
    anchors[year] = float(st.session_state[key])
    unchanged = all(
        abs(anchors[y] - spec.anchors[y].base) <= 1e-9 * max(1.0, abs(spec.anchors[y].base))
        for y in anchors if y in spec.anchors
    )
    if unchanged:
        edits.pop(feature, None)
    st.session_state.whatif_preset = None
    _invalidate_result()


def _apply_preset(name: str) -> None:
    st.session_state.whatif_edits = preset_edits(WHATIF_PRESETS[name], list(_specs().values()))
    st.session_state.whatif_preset = name
    _bump_generation()
    _invalidate_result()


def _reset_levers() -> None:
    st.session_state.whatif_edits = {}
    st.session_state.whatif_preset = None
    _bump_generation()
    _invalidate_result()


# ── Controls ──────────────────────────────────────────────────────────────


def _ordered(specs: List[LeverSpec]) -> List[LeverSpec]:
    """Key levers first, in their configured order, then the rest as listed."""
    by_feature = {spec.feature: spec for spec in specs}
    key = [by_feature[f] for f in WHATIF_KEY_LEVERS if f in by_feature]
    rest = [spec for spec in specs if spec.feature not in WHATIF_KEY_LEVERS]
    return key + rest


def _multiplier_slider(spec: LeverSpec) -> None:
    label = f"{spec.feature} — × by {spec.last_year}"
    key = _widget_key(f"{spec.feature}_mult")
    if spec.multiplier_bounds is None:
        st.slider(label, 0.0, 2.0, 1.0, key=key, disabled=True, help=f"Locked: {spec.multiplier_reason}")
        return
    end = spec.anchors[spec.last_year]
    lo, hi = spec.multiplier_bounds
    current = float(np.clip(multiplier_from_anchors(spec, st.session_state.whatif_edits.get(spec.feature, {})), lo, hi))
    col_slider, col_value = st.columns([4, 1])
    with col_slider:
        st.slider(
            label, min_value=float(lo), max_value=float(hi), value=current, step=0.01, key=key,
            on_change=_on_multiplier_change, args=(spec.feature, key),
            help=(
                f"AR6 5–95% range in {spec.last_year}: {end.band_lo:.4g} to {end.band_hi:.4g}; "
                f"baseline {end.base:.4g}. The path ramps from the end of history to this multiple."
            ),
        )
    with col_value:
        st.caption(f"{spec.last_year}: {end.base:.4g} → {end.base * current:.4g}")


def _render_multiplier_controls(specs: List[LeverSpec]) -> None:
    ordered = _ordered(specs)
    key_levers = [spec for spec in ordered if spec.feature in WHATIF_KEY_LEVERS]
    others = [spec for spec in ordered if spec.feature not in WHATIF_KEY_LEVERS]
    for spec in key_levers:
        _multiplier_slider(spec)
    if others:
        with st.expander(f"All inputs ({len(others)} more)"):
            for spec in others:
                _multiplier_slider(spec)


def _render_anchor_controls(specs: List[LeverSpec], anchor_years: List[int]) -> None:
    edits = st.session_state.whatif_edits
    ordered = _ordered(specs)
    movable = [spec for spec in ordered if spec.enabled]
    locked = [spec for spec in ordered if not spec.enabled]
    if not movable:
        st.info("No input of this trajectory can be moved.")
        return
    by_feature = {spec.feature: spec for spec in movable}
    if st.session_state.get("whatif_adv_feature") not in by_feature:
        st.session_state.whatif_adv_feature = movable[0].feature
    feature = st.selectbox(
        "Lever", list(by_feature), key="whatif_adv_feature",
        format_func=lambda f: f"{f}  (edited)" if f in edits else f,
    )
    spec = by_feature[feature]
    columns = st.columns(min(4, max(1, len(anchor_years))))
    for i, year in enumerate(anchor_years):
        bounds = spec.anchors[year]
        current = float(np.clip(edits.get(feature, {}).get(year, bounds.base), bounds.lo, bounds.hi))
        span = bounds.hi - bounds.lo
        key = _widget_key(f"{feature}_{year}")
        with columns[i % len(columns)]:
            st.slider(
                str(year), min_value=float(bounds.lo), max_value=float(bounds.hi), value=current,
                step=float(span / 200) if span > 0 else 1.0, key=key, format="%.4g",
                on_change=_on_anchor_change, args=(feature, year, key),
                help=f"AR6 5–95%: {bounds.band_lo:.4g} to {bounds.band_hi:.4g}; baseline {bounds.base:.4g}",
            )
    if locked:
        st.caption("Locked: " + "; ".join(f"{spec.feature} ({spec.reason})" for spec in locked))


def _render_controls(specs: List[LeverSpec], anchor_years: List[int]) -> None:
    st.markdown("**Levers**")
    columns = st.columns(len(WHATIF_PRESETS) + 1)
    for column, name in zip(columns, WHATIF_PRESETS):
        column.button(name, key=f"whatif_preset_{name}", on_click=_apply_preset, args=(name,), use_container_width=True)
    columns[-1].button("Reset levers", key="whatif_reset", on_click=_reset_levers, use_container_width=True)
    advanced = st.toggle(
        "Unlock per-decade anchors", key="whatif_advanced",
        help="Pin one lever's value in each decade instead of a single multiple reached at the last year.",
    )
    if advanced:
        _render_anchor_controls(specs, anchor_years)
    else:
        _render_multiplier_controls(specs)


def _render_overlay(specs: List[LeverSpec]) -> None:
    edits = st.session_state.whatif_edits
    st.markdown("**Input trajectories**")
    col_mode, col_pick = st.columns([1, 2])
    with col_mode:
        mode_label = st.radio("Scale", list(OVERLAY_MODES), key="whatif_overlay_mode")
    available = [spec.feature for spec in specs]
    default_pick = [f for f in WHATIF_KEY_LEVERS if f in available] + [f for f in edits if f in available and f not in WHATIF_KEY_LEVERS]
    with col_pick:
        chosen = st.multiselect("Levers shown", available, default=default_pick, key=_widget_key("overlay_pick"))
    shown = list(dict.fromkeys(list(chosen) + [f for f in edits if f in available]))
    fig = plot_lever_overlay(specs, edits, mode=OVERLAY_MODES[mode_label], features=shown)
    st.pyplot(fig)
    plt.close(fig)
    inside, total = band_coverage(edits, specs)
    if total:
        note = "" if inside == total else " Values outside it are where the emulator is least reliable."
        st.caption(f"{inside} of {total} edited lever values lie inside the AR6 5–95% range.{note}")
    else:
        st.caption("No levers edited yet: the emulator will reproduce its forecast of the unchanged inputs.")


# ── Running the emulator ──────────────────────────────────────────────────


def _render_run(run_id, engine, prepared, region, candidate, rows, history_steps, specs, region_r2, on_plot_saved) -> None:
    edits = st.session_state.whatif_edits
    if st.button("Run emulator", type="primary", key="whatif_run"):
        from src.inference.tft_predict import check_vocabulary, predict_windows

        problems = check_vocabulary(engine, rows)
        if problems:
            st.error("The model's vocabulary lacks: " + ", ".join(problems))
            return
        with st.spinner("Emulating…"):
            edited = apply_levers(rows, edits, history_steps)
            pred_baseline = _baseline_prediction(run_id, region, candidate.model, candidate.scenario)
            pred_edited = predict_windows(engine, edited) if edits else pred_baseline
            result = assemble_result(
                run_id, region, candidate, rows, edited, prepared.raw_features, prepared.targets,
                pred_baseline, pred_edited, edits, band_coverage(edits, specs), history_steps,
            )
        st.session_state.whatif_result = result
        metadata = result_metadata(result, float(region_r2), st.session_state.whatif_preset)
        fig = plot_whatif_grid(result, ar6_band=_ar6_band(run_id, region))
        png_path, _ = save_whatif_outputs(run_id, fig, metadata)
        plt.close(fig)
        st.session_state.whatif_saved = png_path
        if on_plot_saved is not None:
            on_plot_saved()
        logging.info("What-if: %s / %s in %s with %d edited lever(s) saved to %s",
                     candidate.model, candidate.scenario, region, len(edits), png_path)

    result = st.session_state.whatif_result
    if result is None:
        return
    show_band = st.checkbox("Shade the 5–95% range of AR6 scenarios in this region", value=True, key="whatif_show_band")
    fig = plot_whatif_grid(result, ar6_band=_ar6_band(run_id, region) if show_band else None)
    st.pyplot(fig)
    plt.close(fig)
    if result.split != "test":
        st.caption(
            "The dashed line is the emulator's fit of a scenario it was trained on, not held-out "
            "accuracy; the region's R² above comes from scenarios it never saw."
        )
    if st.session_state.whatif_saved:
        st.caption(f"Saved as {os.path.basename(st.session_state.whatif_saved)} (see Recent Plots).")
    with st.expander("Emulator vs IAM on the unchanged inputs"):
        st.dataframe(pd.DataFrame([result.baseline_fit]), use_container_width=True)
    st.download_button(
        "Download inputs and forecasts (CSV)",
        export_frame(result).to_csv(index=False).encode("utf-8"),
        file_name=f"whatif_{run_id}_{region}.csv",
        mime="text/csv",
        key="whatif_download",
    )


# ── The view ──────────────────────────────────────────────────────────────


def render_whatif_view(run_id: str, on_plot_saved: Optional[Callable[[], None]] = None) -> None:
    """Render the what-if view for *run_id*.

    *on_plot_saved* is called after a figure lands in the run's saved plots,
    so the caller can refresh its sidebar listing.
    """
    st.subheader(VIEW_NAME)
    st.warning(CAVEAT.format(threshold=WHATIF_R2_THRESHOLD, min_samples=WHATIF_MIN_SAMPLE_SIZE))

    model_type = run_id.split("_", 1)[0]
    if model_type not in WHATIF_ENGINES:
        st.info("Live emulation is available for TFT runs only; the XGB and LSTM engines are not wired up yet.")
        return
    try:
        engine = _engine(run_id)
    except ImportError as exc:
        st.error(
            f"The dashboard environment lacks the deep-learning stack ({exc}). Install it there with "
            "`pip install -r requirements-dashboard.txt`."
        )
        return
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    try:
        prepared = _prepared(run_id)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    table = _region_table(run_id)
    if table is None:
        st.error("This run has no per-region metrics and no saved predictions to compute them from. Run its test phase first.")
        return
    gated = eligible_regions(table, WHATIF_R2_THRESHOLD, WHATIF_MIN_SAMPLE_SIZE)
    offered = [region for region in gated["Region"] if _candidates(run_id, region)]
    if not offered:
        st.error("No region clears the accuracy gate with a usable baseline trajectory.")
        return
    hidden = len(gated) - len(offered)

    _state_default("whatif_gen", 0)
    _state_default("whatif_edits", {})
    _state_default("whatif_preset", None)
    _state_default("whatif_result", None)
    _state_default("whatif_saved", None)

    # Region
    if "whatif_region" not in st.session_state:
        st.session_state.whatif_region = st.session_state.get("whatif_region_choice")
    if st.session_state.get("whatif_region") not in offered:
        st.session_state.whatif_region = WHATIF_DEFAULT_REGION if WHATIF_DEFAULT_REGION in offered else offered[0]
    col_region, col_r2, col_n = st.columns([3, 1, 1])
    with col_region:
        hidden_note = (
            f" {hidden} more clear the gate but have no {WHATIF_BASELINE_CATEGORY} trajectory long enough to emulate."
            if hidden else ""
        )
        region = st.selectbox(
            "Region", offered, key="whatif_region",
            help=f"Regions with pooled R² ≥ {WHATIF_R2_THRESHOLD:.2f} on at least {WHATIF_MIN_SAMPLE_SIZE:,} held-out values.{hidden_note}",
        )
    st.session_state.whatif_region_choice = region
    region_row = gated[gated["Region"] == region].iloc[0]
    with col_r2:
        st.metric("Held-out pooled R²", f"{region_row[R2_COLUMN]:.3f}")
    with col_n:
        st.metric("Scored values", f"{int(region_row[SAMPLE_COLUMN]):,}")

    # Baseline
    candidates = _candidates(run_id, region)
    by_label = {candidate.label: candidate for candidate in candidates}
    n_inputs = len(prepared.raw_features)
    if "whatif_baseline" not in st.session_state:
        st.session_state.whatif_baseline = st.session_state.get("whatif_baseline_choice")
    if st.session_state.get("whatif_baseline") not in by_label:
        st.session_state.whatif_baseline = choose_default_baseline(candidates).label
    candidate = by_label[st.selectbox(
        "Baseline scenario", list(by_label), key="whatif_baseline",
        format_func=lambda label: f"{label} — {by_label[label].n_reported}/{n_inputs} inputs reported · {by_label[label].split} split",
        help=f"{WHATIF_BASELINE_CATEGORY} trajectories in this region with at least {engine.min_steps} steps.",
    )]
    st.session_state.whatif_baseline_choice = candidate.label

    selection = (run_id, region, candidate.key)
    if st.session_state.get("whatif_prev_selection") != selection:
        st.session_state.whatif_prev_selection = selection
        st.session_state.whatif_edits = {}
        st.session_state.whatif_preset = None
        _bump_generation()
        _invalidate_result()

    rows = baseline_rows(prepared, region, candidate.model, candidate.scenario)
    history_steps = engine.encoder_length
    years = [int(y) for y in rows["Year"]]
    history_years = years[:history_steps]
    anchor_years = resolve_anchor_years(years, history_years)
    specs = build_lever_specs(rows, _bands(run_id, region), prepared.raw_features, history_steps, anchor_years)
    st.session_state.whatif_specs = {spec.feature: spec for spec in specs}
    st.caption(
        f"History {history_years[0]}–{history_years[-1]} stays as the IAM reported it; the emulator forecasts "
        f"{years[history_steps]}–{years[-1]}. {candidate.n_reported} of {n_inputs} inputs were reported by "
        f"the IAM for this trajectory; the rest are imputed and locked."
    )

    _render_controls(specs, anchor_years)
    _render_overlay(specs)
    _render_run(run_id, engine, prepared, region, candidate, rows, history_steps, specs, region_row[R2_COLUMN], on_plot_saved)
