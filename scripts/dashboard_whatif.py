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
    WHATIF_COMBINATION_DEFAULTS,
    WHATIF_COMPARISON_OUTPUTS,
    WHATIF_DISTRIBUTION_YEAR,
    WHATIF_DEFAULT_REGION,
    WHATIF_ENGINES,
    WHATIF_KEY_LEVERS,
    WHATIF_MIN_SAMPLE_SIZE,
    WHATIF_MAX_COMBINATION_LEVERS,
    WHATIF_PRESETS,
    WHATIF_R2_THRESHOLD,
)
from src.inference.whatif import (
    WHATIF_CATEGORY_GROUPS,
    R2_COLUMN,
    SAMPLE_COLUMN,
    LeverSpec,
    apply_levers,
    ar6_target_bands,
    ar6_distribution_samples,
    assemble_result,
    band_coverage,
    baseline_rows,
    build_lever_specs,
    candidate_baselines,
    comparison_sources,
    choose_default_baseline,
    eligible_regions,
    export_frame,
    high_low_combinations,
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
from src.visualization.whatif import plot_lever_overlay, plot_whatif_grid, plot_whatif_comparison, save_whatif_outputs

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
    from src.inference.engines import load_engine

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
    if all(state.get(key) is not None for key in ("test_data", "y_test", "preds", "targets")):
        return region_metrics_from_predictions(run_id, state.test_data, state.y_test, state.preds, state.targets)
    return None


@st.cache_data(show_spinner=False)
def _candidates(run_id: str, region: str):
    return candidate_baselines(_prepared(run_id), region, min_steps=_engine(run_id).min_steps)


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


@st.cache_data(show_spinner="Selecting source scenarios…")
def _comparison_sources(run_id, region, history_steps, min_steps, preferred):
    return comparison_sources(
        _prepared(run_id), region, _bands(run_id, region),
        WHATIF_COMBINATION_DEFAULTS, history_steps, min_steps, preferred,
    )


@st.cache_data(show_spinner="Emulating the unchanged inputs…")
def _baseline_prediction(run_id: str, region: str, model: str, scenario: str) -> pd.DataFrame:
    from src.inference.engines import predict_windows

    rows = baseline_rows(_prepared(run_id), region, model, scenario)
    return predict_windows(_engine(run_id), rows)


# ── Session state ─────────────────────────────────────────────────────────


def _state_default(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _invalidate_result() -> None:
    st.session_state.whatif_result = None
    st.session_state.whatif_saved = None
    st.session_state.whatif_combinations = None


def _bump_generation() -> None:
    """Re-create every lever widget on the next run, so it reads the new edits."""
    st.session_state.whatif_gen = st.session_state.get("whatif_gen", 0) + 1


def _widget_key(name: str) -> str:
    run_id = st.session_state.get("current_run_id", "preview")
    return f"whatif_w_{run_id}_{st.session_state.get('whatif_gen', 0)}_{name}"


def _specs() -> Dict[str, LeverSpec]:
    return st.session_state.whatif_specs


def _on_multiplier_change(feature: str, key: str) -> None:
    # A delayed browser event can arrive after switching runs cleared these.
    if (key != _widget_key(f"{feature}_mult")
            or feature not in st.session_state.get("whatif_specs", {})
            or key not in st.session_state or "whatif_edits" not in st.session_state):
        return
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
    if (key != _widget_key(f"{feature}_{year}")
            or feature not in st.session_state.get("whatif_specs", {})
            or key not in st.session_state or "whatif_edits" not in st.session_state):
        return
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
    if not st.session_state.get("whatif_specs") or "whatif_edits" not in st.session_state:
        return
    specs = list(_specs().values())
    # Each pair controls one input. Preserve choices made in the other pairs.
    st.session_state.whatif_edits.update(preset_edits(WHATIF_PRESETS[name], specs))
    selected = []
    for label, preset in WHATIF_PRESETS.items():
        values = preset_edits(preset, specs)
        if values and all(st.session_state.whatif_edits.get(f) == anchors for f, anchors in values.items()):
            selected.append(label)
    st.session_state.whatif_preset = "; ".join(selected) or None
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
    specs = [spec for spec in specs if spec.feature != "GDP|PPP"]
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
    st.caption("Choose High or Low independently for each input. Run emulator uses your current choices; Run all High/Low combinations tests all 16 combinations per available baseline group.")
    by_feature = {spec.feature: spec for spec in specs}
    names = list(WHATIF_PRESETS)
    with st.container(key="whatif_preset_grid"):
        for start in range(0, len(names), 2):
            for column, name in zip(st.columns(2), names[start:start + 2]):
                available = all(f in by_feature and by_feature[f].enabled for f in WHATIF_PRESETS[name])
                values = preset_edits(WHATIF_PRESETS[name], specs) if available else {}
                active = bool(values) and all(st.session_state.whatif_edits.get(f) == a for f, a in values.items())
                column.button(
                    name, key=f"whatif_preset_{name}", on_click=_apply_preset, args=(name,),
                    use_container_width=True, disabled=not available,
                    type="primary" if active else "secondary",
                    help=None if available else "This baseline does not have a movable input for this preset.",
                )
    st.button("Reset levers", key="whatif_reset", on_click=_reset_levers)
    st.caption("GDP MER controls both GDP inputs. PPP follows the same relative change at each forecast year; source PPP/MER ratios and historical values are preserved.")

    advanced = st.toggle(
        "Unlock per-decade anchors", key="whatif_advanced",
        help="Pin one lever's value in each decade instead of a single multiple reached at the last year.",
    )
    if advanced:
        _render_anchor_controls(specs, anchor_years)
    else:
        _render_multiplier_controls(specs)


def _render_gdp_values(rows, history_steps):
    if not {"GDP|MER", "GDP|PPP"} <= set(rows.columns):
        return
    edited = apply_levers(rows, st.session_state.whatif_edits, history_steps)
    with st.expander("GDP values · PPP follows MER", expanded=False):
        st.caption("Read-only values in the source scenario's units, before model scaling. PPP = source PPP × edited MER / source MER. When source MER is zero, PPP stays unchanged and the skip is logged.")
        st.dataframe(pd.DataFrame({
            "Year": rows["Year"].to_numpy(),
            "Period": ["Fixed history" if i < history_steps else "Forecast" for i in range(len(rows))],
            "MER · source": rows["GDP|MER"].to_numpy(),
            "MER · edited": edited["GDP|MER"].to_numpy(),
            "PPP · source": rows["GDP|PPP"].to_numpy(),
            "PPP · derived": edited["GDP|PPP"].to_numpy(),
        }), hide_index=True, use_container_width=True)


def _render_overlay(specs: List[LeverSpec]) -> None:
    edits = st.session_state.whatif_edits
    st.markdown("**Input trajectories**")
    col_mode, col_pick = st.columns([1, 2])
    with col_mode:
        mode_label = st.radio("Scale", list(OVERLAY_MODES), key="whatif_overlay_mode")
    available = [spec.feature for spec in specs if spec.feature != "GDP|PPP"]
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
        from src.inference.engines import check_vocabulary, predict_windows

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


def _render_combinations(run_id, engine, prepared, region, candidate, rows, history_steps, specs, region_r2, on_plot_saved):
    st.markdown("**High/Low combinations**")
    chosen = list(WHATIF_COMBINATION_DEFAULTS)
    year = WHATIF_DISTRIBUTION_YEAR
    outputs = [t for t in WHATIF_COMPARISON_OUTPUTS if t in prepared.targets]
    sources, unavailable = _comparison_sources(run_id, region, history_steps, engine.min_steps, candidate.key)
    if unavailable:
        st.info("No source with all four movable inputs in: " + ", ".join(unavailable) + ". These groups are not included.")
    if not sources:
        return
    per_group = 2 ** len(chosen)
    count = per_group * len(sources)
    max_levers = WHATIF_MAX_COMBINATION_LEVERS
    st.caption(
        f"All {per_group} High/Low combinations for each baseline group: {count} paths plus the original emulations. "
        "The panels on the right compare AR6 and generated values in 2050."
    )
    st.caption("Each run uses the full model input set. Carbon price, solar cost, population and GDP MER are the four independent factors. PPP follows MER proportionally at each forecast year; other inputs retain their starting scenario's values.")
    with st.expander("How to read this chart", expanded=False):
        st.caption("This panel only explains the chart. Opening it does not change any settings or results.")
        st.caption(
            "**Left: how the paths change over time**\n\n"
            "- **Thin lines:** all 16 High/Low combinations of carbon price, solar cost, population and GDP MER for each starting scenario.\n"
            "- **Dashed line:** the emulator's result with that starting scenario's inputs unchanged.\n"
            "- **Filled area:** the minimum to maximum generated value at each year. It is not a confidence interval.\n"
            "- **Color shades:** the starting scenario's AR6 category group (C1–4, C5–6 or C7–8). Overlapping lines and areas look darker; darkness is not a probability.\n\n"
            "**Right: compare the outcomes in 2050**\n\n"
            "- **AR6, left box:** reported values from the original scenarios in that group and region.\n"
            "- **Synthetic, right box:** values from the generated High/Low combinations.\n"
            "- **Black diamond:** the unchanged emulation in 2050. Each box shows the median and middle 50%; whiskers span the 5th–95th percentiles.\n\n"
            "The group labels describe the starting scenarios. The generated paths have not been assigned new climate categories."
        )
    with st.expander("View data sources", expanded=False):
        st.write("These are the starting scenarios used to generate the colored paths. They are chosen automatically; there is nothing to select here.")
        if run_id == "preview":
            st.info("Preview only: the model and scenario names below are fictional examples, not real AR6 sources.")
        st.dataframe(pd.DataFrame([
            {"Chart group": group, "Source IAM": selected.model, "Starting scenario": selected.scenario,
             "Generated paths": per_group}
            for group, (selected, _, _) in sources.items()
        ]), hide_index=True, use_container_width=True)
        st.caption("One source per group must report all four inputs and have enough data for the emulator. PPP follows MER; other inputs retain their source values. Source names are provided so you can trace where the chart starts, not as settings you need to choose.")
    signature = (run_id, region, tuple((g, c.key) for g, (c, _, _) in sources.items()), tuple(chosen))
    saved = st.session_state.get("whatif_combinations")
    if saved is not None and (saved["signature"] != signature or "ensembles" not in saved):
        st.session_state.whatif_combinations = None
        saved = None
    if st.button("Run all High/Low combinations", key="whatif_run_combinations", type="primary", disabled=not chosen or not outputs or not sources):
        from src.inference.engines import check_vocabulary, predict_windows

        problems = [f"{group}: {problem}" for group, (_, source_rows, _) in sources.items() for problem in check_vocabulary(engine, source_rows)]
        if problems:
            st.error("The model's vocabulary lacks: " + ", ".join(problems))
            return
        ensembles, exports, records = {}, [], []
        progress = st.progress(0.0, text=f"Emulating 0 of {count} combinations…")
        try:
            completed = 0
            for group, (selected, source_rows, source_specs) in sources.items():
                baseline_pred = _baseline_prediction(run_id, region, selected.model, selected.scenario)
                original = assemble_result(
                    run_id, region, selected, source_rows, source_rows, prepared.raw_features, prepared.targets,
                    baseline_pred, baseline_pred, {}, (0, 0), history_steps,
                )
                exports.append(export_frame(original).assign(combination="Original emulation", baseline_group=group,
                                                             baseline_model=selected.model, baseline_scenario=selected.scenario))
                predictions = {}
                for label, edits in high_low_combinations(source_specs, chosen, max_levers=max_levers):
                    edited = apply_levers(source_rows, edits, history_steps)
                    prediction = predict_windows(engine, edited)
                    result = assemble_result(
                        run_id, region, selected, source_rows, edited, prepared.raw_features, prepared.targets,
                        baseline_pred, prediction, edits, band_coverage(edits, source_specs), history_steps,
                    )
                    predictions[label] = result.pred_edited
                    exports.append(export_frame(result).assign(combination=label, baseline_group=group,
                                                                baseline_model=selected.model, baseline_scenario=selected.scenario))
                    records.append({"group": group, "label": label, "result": result_metadata(result, float(region_r2), None)})
                    completed += 1
                    progress.progress(completed / count, text=f"Emulating {completed} of {count} combinations…")
                ensembles[group] = {"original": original, "predictions": predictions}
            metadata = result_metadata(next(iter(ensembles.values()))["original"], float(region_r2), None)
            metadata.update({"mode": "high_low_comparison", "features": chosen, "combinations": records,
                             "distribution_year": year, "display_targets": outputs, "grouping": "source baseline category"})
            reference = ar6_distribution_samples(prepared.frame, region, prepared.targets, year)
            metadata["reference_samples"] = reference.to_dict(orient="records")
            fig = plot_whatif_comparison(ensembles, reference, year=year, targets=outputs)
            try:
                png_path, _ = save_whatif_outputs(run_id, fig, metadata)
            finally:
                plt.close(fig)
            saved = {"signature": signature, "ensembles": ensembles,
                     "csv": pd.concat(exports, ignore_index=True).to_csv(index=False).encode("utf-8"),
                     "png_path": png_path}
            st.session_state.whatif_combinations = saved
            if on_plot_saved is not None:
                on_plot_saved()
        except Exception as exc:
            logging.exception("High/Low combination run failed")
            st.error(f"The combination run did not finish: {exc}. No partial ensemble was saved.")
        finally:
            progress.empty()
    if saved is None:
        return
    if not outputs:
        st.info("Choose at least one output to display.")
        return
    reference = ar6_distribution_samples(prepared.frame, region, prepared.targets, year)
    fig = plot_whatif_comparison(saved["ensembles"], reference, year=year, targets=outputs)
    st.pyplot(fig)
    plt.close(fig)
    total = sum(len(bundle["predictions"]) for bundle in saved["ensembles"].values())
    st.caption(f"All {total} combinations completed. Initial figure saved as {os.path.basename(saved['png_path'])}.")
    st.download_button(
        "Download all combination inputs and forecasts (CSV)", saved["csv"],
        file_name=f"whatif_combinations_{run_id}_{region}.csv", mime="text/csv", key="whatif_combinations_download",
    )
    st.download_button("Download AR6 distribution samples (CSV)", reference.to_csv(index=False).encode("utf-8"),
                       file_name=f"whatif_ar6_{region}_{year}.csv", mime="text/csv", key="whatif_ar6_download")


def render_whatif_view(run_id: str, on_plot_saved: Optional[Callable[[], None]] = None) -> None:
    """Render the what-if view for *run_id*.

    *on_plot_saved* is called after a figure lands in the run's saved plots,
    so the caller can refresh its sidebar listing.
    """
    st.subheader(VIEW_NAME)
    st.warning(CAVEAT.format(threshold=WHATIF_R2_THRESHOLD, min_samples=WHATIF_MIN_SAMPLE_SIZE))

    model_type = run_id.split("_", 1)[0]
    if model_type not in WHATIF_ENGINES:
        st.info("Live emulation is available for TFT, LSTM and XGB runs.")
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

    if st.session_state.get("whatif_gdp_linkage") != "source-ratio-v1":
        _invalidate_result()
        st.session_state.whatif_gdp_linkage = "source-ratio-v1"
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
    # A browser session opened before this revision may still contain PPP edits.
    if "GDP|PPP" in st.session_state.whatif_edits:
        st.session_state.whatif_edits.pop("GDP|PPP")
        _bump_generation()
        _invalidate_result()
    st.caption(
        f"History {history_years[0]}–{history_years[-1]} stays as the IAM reported it; the emulator forecasts "
        f"{years[history_steps]}–{years[-1]}. {candidate.n_reported} of {n_inputs} inputs were reported by "
        f"the IAM for this trajectory; the rest are imputed and locked."
    )

    _render_controls(specs, anchor_years)
    _render_gdp_values(rows, history_steps)
    _render_overlay(specs)
    _render_run(run_id, engine, prepared, region, candidate, rows, history_steps, specs, region_row[R2_COLUMN], on_plot_saved)
    _render_combinations(run_id, engine, prepared, region, candidate, rows, history_steps, specs, region_row[R2_COLUMN], on_plot_saved)
