import streamlit as st
import numpy as np
import pandas as pd
import logging
import json
import ast
import os
import sys
import argparse

from src.data.process_data import SCENARIO_CATEGORY_CSV, relabel_scenario_categories
from src.visualization.helpers import backfill_group_labels
from src.visualization.trajectories import plot_trajectories, get_saved_plots_metadata
from src.utils.utils import setup_logging
from src.utils.regions import regions_ordered_by_scale
from src.utils.run_store import RunStore
from configs.data import REGION_CODE_TO_LABEL
from configs.dashboard import DEFAULT_RUNS
import datetime

st.set_page_config(layout="wide")

# Apply global styling for wider sidebar
st.markdown("""
<style>
    @media (min-width: 769px) {
        .css-1d391kg, [data-testid="stSidebar"] {
            width: 25rem !important;
            min-width: 25rem !important;
        }
        .css-1d391kg > div {
            width: 25rem !important;
            min-width: 25rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_unique_values(test_data):
    """Cache unique values for filters."""
    scenario_categories = test_data['Scenario_Category'].unique()

    # Countries first, then progressively coarser aggregates.
    new_region_order = regions_ordered_by_scale(
        test_data['Region'].dropna().astype(str).unique()
    )

    model_families = test_data['Model_Family'].unique()
    return scenario_categories, new_region_order, model_families

@st.cache_data
def load_plot_image(plot_path):
    """Cache the loading of plot images to improve performance."""
    from PIL import Image
    return Image.open(plot_path)

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_cached_saved_plots(run_id):
    """Cache saved plots metadata to improve sidebar performance."""
    return get_saved_plots_metadata(run_id)

def delete_saved_plot(plot_info):
    """Delete a saved plot and its metadata files."""
    try:
        # Delete the plot image file
        if os.path.exists(plot_info['plot_path']):
            os.remove(plot_info['plot_path'])
        
        # Delete the metadata file
        metadata_path = plot_info['plot_path'].replace('.png', '_metadata.json')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        return True
    except Exception as e:
        st.error(f"Error deleting plot: {e}")
        return False


def _present(defaults, options):
    """The defaults a multiselect may show: Streamlit rejects one not in *options*."""
    available = set(options)
    return [value for value in defaults if value in available]


def make_filters(test_data):
    scenario_categories, regions, model_families = get_unique_values(test_data)

    st.session_state.selected_scenario_categories = st.multiselect(
        "Select Scenario Categories",
        options=scenario_categories,
        default=_present(["C3"], scenario_categories),
    )
    st.session_state.selected_regions = st.multiselect(
        "Select Regions", options=regions, default=_present(["World"], regions)
    )
    st.session_state.selected_model_families = st.multiselect(
        "Select Model Families", options=model_families, default=model_families.tolist()
    )

    if st.button("Make New Plot"):
        st.session_state.apply_filters_clicked = True

def apply_filters():
    logging.info("Applying filters to test data...")
    preds = st.session_state.get('preds')
    if preds is None:
        st.error("No predictions found in session state.")
        return

    # TFT/LSTM case: use horizon subset when available (predictions cover forecast horizon only)
    horizon_df = st.session_state.get('horizon_df')
    horizon_y_true = st.session_state.get('horizon_y_true')
    if horizon_df is not None and horizon_y_true is not None:
        test_data = horizon_df
        y_test = horizon_y_true
    # XGBoost / generic case: use full test split
    elif st.session_state.get('y_test') is not None:
        y_test = st.session_state.y_test
        test_data = st.session_state.test_data
    else:
        st.error("Required data not found in session state. Please ensure the model has been trained.")
        return

    # Build filter mask.  A filter whose column the rows lack cannot apply;
    # say so rather than silently showing every value.
    selections = {
        'Scenario_Category': st.session_state.selected_scenario_categories,
        'Region': st.session_state.selected_regions,
        'Model_Family': st.session_state.selected_model_families,
    }
    missing = [column for column in selections if column not in test_data.columns]
    if missing:
        st.warning(
            f"Cannot filter on {', '.join(missing)}: the run's prediction rows carry "
            "no such column, so every value is shown."
        )
    mask_td = pd.Series(True, index=test_data.index)
    for column, selected in selections.items():
        if column in test_data.columns:
            mask_td &= test_data[column].isin(selected)

    # Create target mask and store data
    selected_positions = np.where(mask_td)[0]
    mask_targets = np.zeros(len(y_test), dtype=bool)
    mask_targets[selected_positions] = True
    
    st.session_state.test_mask = mask_td
    st.session_state.target_mask = mask_targets
    st.session_state.current_y_test = y_test
    st.session_state.current_preds = preds
    st.session_state.current_test_data = test_data

def filter_and_plot(run_id):
    filtered_y_test = st.session_state.current_y_test[st.session_state.target_mask]
    filtered_preds = st.session_state.current_preds[st.session_state.target_mask]
    
    # Use the same test_data that was used to create the mask
    current_test_data = st.session_state.get('current_test_data', st.session_state.test_data)
    filtered_test_data = current_test_data[st.session_state.test_mask].reset_index(drop=True)

    # Compute aggregated metrics once (will also be cached) and embed in metadata for persistence
    metrics_df = _compute_filtered_metrics(filtered_y_test, filtered_preds, st.session_state.targets)
    metrics_row = metrics_df.iloc[0].to_dict()
    # Sanitize NaN / Inf for JSON compatibility
    for k, v in list(metrics_row.items()):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            metrics_row[k] = None

    # Prepare filter metadata (including metrics)
    filter_metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'scenario_categories': st.session_state.selected_scenario_categories,
        'regions': st.session_state.selected_regions,
        'model_families': st.session_state.selected_model_families,
        'num_data_points': len(filtered_test_data),
        'targets': st.session_state.targets,
        'metrics': metrics_row
    }

    # `make dashboard` sets these; see the Makefile's SAVE_PLOTS.
    save_individual = os.getenv('SAVE_INDIVIDUAL_PLOTS', 'false').lower() == 'true'
    individual_indices = _individual_plot_indices() if save_individual else []
    logging.debug("save_individual=%s individual_indices=%s", save_individual, individual_indices)

    plot_trajectories(
        filtered_test_data,
        filtered_y_test,
        filtered_preds,
        st.session_state.targets,
        alpha=0.5,
        linewidth=0.5,
        run_id=run_id,
        filter_metadata=filter_metadata,
        save_individual=save_individual,
        individual_indices=individual_indices
    )

    # Metrics expander (appears after plot render)
    with st.expander("View Metrics", expanded=True):
        st.dataframe(metrics_df, use_container_width=True)


def _individual_plot_indices() -> list:
    """Indices from INDIVIDUAL_PLOT_INDICES, e.g. "[0, 6]" or "6"; [0] when unparsable."""
    raw = os.getenv('INDIVIDUAL_PLOT_INDICES', '[0]')
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [0]
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]
    try:
        return [int(x) for x in parsed]
    except (TypeError, ValueError):
        return [0]


@st.cache_data(show_spinner=False)
def _compute_filtered_metrics(y_true_filtered: np.ndarray, y_pred_filtered: np.ndarray, targets):
    """Compute aggregate metrics across all targets for filtered instances.

    Flattens (n_samples, n_targets) arrays after masking NaNs per element.
    Returns single-row DataFrame.
    """
    y_true = np.asarray(y_true_filtered)
    y_pred = np.asarray(y_pred_filtered)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    # Align columns
    n_targets = min(y_true.shape[1], y_pred.shape[1])
    y_true = y_true[:, :n_targets]
    y_pred = y_pred[:, :n_targets]
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    yt_flat = y_true[mask]
    yp_flat = y_pred[mask]
    count = yt_flat.size
    if count == 0:
        return pd.DataFrame([{'Count': 0, 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE_%': np.nan}])
    var = np.var(yt_flat)
    if var == 0:
        r2 = np.nan
    else:
        ss_res = np.sum((yt_flat - yp_flat) ** 2)
        ss_tot = np.sum((yt_flat - yt_flat.mean()) ** 2)
        r2 = np.nan if ss_tot == 0 else 1 - ss_res/ss_tot
    mae = float(np.mean(np.abs(yt_flat - yp_flat)))
    rmse = float(np.sqrt(np.mean((yt_flat - yp_flat) ** 2)))
    non_zero = np.abs(yt_flat) > 1e-12
    if non_zero.any():
        mape = float(np.mean(np.abs((yt_flat[non_zero] - yp_flat[non_zero]) / yt_flat[non_zero])) * 100)
    else:
        mape = np.nan
    return pd.DataFrame([{'Count': int(count), 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE_%': mape}])

def display_recent_plots_sidebar(run_id):
    """Display recent plots in the sidebar."""
    st.sidebar.subheader("Recent Plots")
    st.sidebar.markdown("---")
    
    saved_plots = get_cached_saved_plots(run_id)
    if saved_plots:
        for i, plot_info in enumerate(saved_plots[:10]):  # Show last 10 plots
            metadata = plot_info['metadata']
            timestamp = datetime.datetime.fromisoformat(metadata['timestamp']).strftime('%Y-%m-%d %H:%M')
            
            # Create a container for each plot with markdown + button
            with st.sidebar.container():
                # Create columns for metadata and delete button
                col1, col2 = st.sidebar.columns([6, 1])
                
                with col1:
                    # Display metadata as markdown
                    metadata_text = f"**{timestamp}**"
                    
                    if metadata.get('scenario_categories'):
                        scenarios = metadata['scenario_categories']
                        metadata_text += f"  \n📈  {', '.join(scenarios)}"
                    
                    if metadata.get('regions'):
                        regions = metadata['regions']
                        metadata_text += f"  \n🌍  {', '.join(regions)}"
                    
                    if metadata.get('model_families'):
                        models = metadata['model_families']
                        metadata_text += f"  \n🤖  {', '.join(models)}"
                    
                    metadata_text += f"  \n{metadata.get('num_data_points', 0)} points"

                    # Show quick metrics summary if available
                    metrics = metadata.get('metrics')
                    if metrics:
                        r2_disp = metrics.get('R2')
                        rmse_disp = metrics.get('RMSE')
                        mae_disp = metrics.get('MAE')
                        # Format numbers if not None
                        def _fmt(v):
                            if v is None or (isinstance(v, float) and np.isnan(v)):
                                return '—'
                            return f"{v:.3f}"
                        metadata_text += (
                            f"  \nR2 {_fmt(r2_disp)} | RMSE {_fmt(rmse_disp)} | MAE {_fmt(mae_disp)}"
                        )
                    
                    st.markdown(metadata_text)
                
                with col2:
                    # Position delete button at the right
                    st.write("")  # Empty space to align delete button
                    st.write("")
                    st.write("")
                    if st.button("✕", key=f"delete_{i}", help="Delete this plot"):
                        if delete_saved_plot(plot_info):
                            # Clear cache to update the list
                            get_cached_saved_plots.clear()
                            # If this was the selected plot, clear selection
                            if st.session_state.get("selected_plot") == plot_info:
                                st.session_state.selected_plot = None
                            st.rerun()
                
                # View plot button spans full width
                if st.sidebar.button("View Plot", key=f"plot_{i}"):
                    st.session_state.selected_plot = plot_info
                
                st.sidebar.markdown("---")  # Separator between plots
    else:
        st.sidebar.write("No saved plots yet.")

def display_selected_plot():
    """Display the selected plot with metadata."""
    if st.session_state.get("selected_plot") is not None:
        if st.button("Hide Recent Plots", key="clear_plot"):
            st.session_state.selected_plot = None
            # Force early exit to prevent showing content after clearing
        else:
            # Only show plot content if clear button wasn't clicked
            plot_info = st.session_state.selected_plot
            st.subheader("Previously Saved Plot")
            
            # Show filter conditions
            metadata = plot_info['metadata']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Scenario Categories:**")
                st.write(", ".join(metadata.get('scenario_categories', [])))
            
            with col2:
                st.write("**Regions:**")
                st.write(", ".join(metadata.get('regions', [])))
            
            with col3:
                st.write("**Model Families:**")
                st.write(", ".join(metadata.get('model_families', [])))
            
            st.write(f"**Data Points:** {metadata.get('num_data_points', 0)}")
            st.write(f"**Created:** {datetime.datetime.fromisoformat(metadata['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Display the plot using cached loading
            img = load_plot_image(plot_info['plot_path'])
            st.image(img, caption="Temporal trajectories", use_container_width=True)

def _code_maps(store):
    """code -> label maps for this run, from the vocabularies it saved.

    The vocabularies have to come from the run: this process never runs
    preprocessing, so the configs.data globals are empty here and mapping
    through them turns every region into NaN.
    """
    maps = {}
    if store.has_categories():
        for column, labels in store.load_categories().items():
            maps[column] = {i: label for i, label in enumerate(labels)}

    # Runs made before categories.json existed.
    if 'Model_Family' not in maps and store.has_train_meta():
        legacy = store.load_train_meta().get('lstm_model_family_categories')
        if legacy:
            maps['Model_Family'] = {i: label for i, label in enumerate(legacy)}
    if 'Region' not in maps and REGION_CODE_TO_LABEL:
        maps['Region'] = dict(REGION_CODE_TO_LABEL)

    return maps


# Per-group labels the dashboard filters on that a saved horizon frame may lack.
LABEL_COLUMNS = ['Scenario_Category', 'Model_Family']


def _backfill_horizon_labels(session_state):
    """Give TFT horizon rows the labels the filters read, from test_data."""
    horizon_df = session_state.get('horizon_df')
    test_data = session_state.get('test_data')
    if horizon_df is None or test_data is None:
        return
    horizon_df, filled = backfill_group_labels(horizon_df, test_data, LABEL_COLUMNS)
    if filled:
        logging.info("horizon_df: copied %s from test_data", ", ".join(filled))
        session_state.horizon_df = horizon_df


def _relabel_scenario_categories(session_state):
    """Label runs by (Model, Scenario) from the metadata table.

    Runs preprocessed while metadata/scenario_category.csv was keyed by
    scenario name alone stored another model's category for about a fifth of
    the runs; the column is not a feature, so the stored labels can be
    corrected here.
    """
    scenario_cat = pd.read_csv(SCENARIO_CATEGORY_CSV, dtype=str)
    for attr in ('test_data', 'horizon_df'):
        df = session_state.get(attr)
        if df is None or not {'Model', 'Scenario', 'Scenario_Category'} <= set(df.columns):
            continue
        df, n_changed = relabel_scenario_categories(df, scenario_cat)
        if n_changed:
            logging.info("%s: %d runs relabelled from %s", attr, n_changed, SCENARIO_CATEGORY_CSV.name)
            setattr(session_state, attr, df)


def _decode_categorical_columns(store, session_state):
    """Decode integer-encoded Region/Model_Family columns back to string labels."""
    maps = _code_maps(store)
    if not maps:
        logging.warning(
            "Run %s has no saved category vocabularies; integer-coded columns "
            "cannot be decoded. Re-run the preprocess phase to write them.",
            store.run_id,
        )
        return

    # Decode in all DataFrames that the dashboard uses for filtering
    for attr in ('test_data', 'horizon_df'):
        df = getattr(session_state, attr, None)
        if df is None:
            continue
        for column, code_map in maps.items():
            if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                continue
            decoded = df[column].map(code_map)
            unmapped = int(decoded.isna().sum() - df[column].isna().sum())
            if unmapped > 0:
                logging.warning(
                    "%s: %d/%d %s codes fell outside the saved vocabulary",
                    attr, unmapped, len(df), column,
                )
            df[column] = decoded


def setup_session_and_logging(run_id):
    """Initialize logging and load run artifacts via RunStore."""
    # Reset state if run_id changed (e.g. via URL query param)
    if st.session_state.get("current_run_id") != run_id:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.current_run_id = run_id

    if st.session_state.get("logging_initialized", False) is False:
        setup_logging(run_id)
        st.session_state.logging_initialized = True

    store = RunStore(run_id)

    if not st.session_state.get("data_initialized"):
        # Load test_data + y_test from pre-saved artifacts (fast path)
        if store.has_test_data():
            test_data, y_test = store.load_test_data()
            features, targets = store.load_features()
            st.session_state.test_data = test_data
            st.session_state.y_test = y_test
            st.session_state.features = features
            st.session_state.targets = targets
        else:
            st.error(
                "No test data artifacts found. Re-run the test phase to generate them:\n\n"
                "Set `run_id` and `resume: test` in a run config, then\n\n"
                "`make train RUN=configs/runs/<your_config>.yaml`\n\n"
                f"(run_id: `{run_id}`)"
            )
            return None

        # Load predictions
        if store.has_predictions():
            pred_bundle = store.load_predictions()
            st.session_state.preds = pred_bundle["preds"]
            if "horizon_df" in pred_bundle:
                st.session_state.horizon_df = pred_bundle["horizon_df"]
            if "horizon_y_true" in pred_bundle:
                st.session_state.horizon_y_true = pred_bundle["horizon_y_true"]
        else:
            st.warning("No predictions found. Run the test phase first.")

        # Decode integer-encoded categoricals (LSTM stores Region/Model_Family as int codes)
        _decode_categorical_columns(store, st.session_state)
        _backfill_horizon_labels(st.session_state)
        _relabel_scenario_categories(st.session_state)

        st.session_state.data_initialized = True

    return True

def handle_filtering_and_plotting(run_id):
    """Handle the filter application and plotting logic."""
    if st.session_state.get("apply_filters_clicked", False):
        apply_filters()
        if st.session_state.get('target_mask') is None:
            return
        if st.session_state.target_mask.sum() == 0:
            st.warning("No data selected with the current filters.")
        else:
            filter_and_plot(run_id)
        st.session_state.apply_filters_clicked = False



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML-IAM Emulation Viewer", add_help=False)
    parser.add_argument("--run_id", "-r", dest="run_id", help="Default run_id to load results from", required=False, default="xgb")
    # Parse known args to ignore Streamlit's own args
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

def resolve_run_id() -> str:
    # CLI run_id as fallback; allow URL ?run_id= to override; reflect final value
    args = parse_args()
    run_id = st.query_params.get("run_id") or args.run_id

    # Resolve model-type shortcuts (e.g. "xgb" → "xgb_76")
    run_id = DEFAULT_RUNS.get(run_id, run_id)

    # Reflect chosen run_id in URL for bookmarking
    st.query_params["run_id"] = run_id
    return run_id

def main():
    run_id = resolve_run_id()
    
    # Initialize session and logging
    session_state = setup_session_and_logging(run_id)
    if not session_state:
        return

    # Main dashboard UI
    st.title("ML-IAM Emulation Viewer")

    # Model selector tabs
    model_type = run_id.split("_", 1)[0]
    cols = st.columns(len(DEFAULT_RUNS))
    for i, (key, default_id) in enumerate(DEFAULT_RUNS.items()):
        label = key.upper()
        with cols[i]:
            if st.button(label, key=f"nav_{key}", use_container_width=True, disabled=(key == model_type)):
                st.query_params["run_id"] = key
                st.rerun()

    make_filters(st.session_state.test_data)
    
    # Handle filtering and plotting
    handle_filtering_and_plotting(run_id)
    
    # Recent plots section
    display_recent_plots_sidebar(run_id)
    
    # Display selected plot section
    display_selected_plot()

if __name__ == "__main__":
    main()