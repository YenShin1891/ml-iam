import streamlit as st
import numpy as np
import logging

from src.utils.plotting import plot_time_series
from src.utils.utils import setup_logging, load_session_state

@st.cache_data
def get_unique_values(test_data):
    """Cache unique values for filters."""
    scenario_categories = test_data['Scenario_Category'].unique()
    
    regions = test_data['Region'].unique()
    R10 = [region for region in regions if region.startswith('R10')]
    R6 = [region for region in regions if region.startswith('R6')]
    R5 = [region for region in regions if region.startswith('R5')]
    World = [region for region in regions if region.startswith('World')]
    other_columns = [region for region in regions if not (region.startswith('R10') or region.startswith('R6') or region.startswith('R5') or region.startswith('World'))]
    new_region_order = other_columns + R10 + R6 + R5 + World
    
    model_families = test_data['Model_Family'].unique()
    return scenario_categories, new_region_order, model_families


def make_filters(test_data):
    scenario_categories, regions, model_families = get_unique_values(test_data)
    
    selected_scenario_categories = st.multiselect(
        "Select Scenario Categories", options=scenario_categories, default=["C3"]
    )
    if selected_scenario_categories != st.session_state.get("selected_scenario_categories", []):
        st.session_state.selected_scenario_categories = selected_scenario_categories

    selected_regions = st.multiselect(
        "Select Regions", options=regions, default=["World"]
    )
    if selected_regions != st.session_state.get("selected_regions", []):
        st.session_state.selected_regions = selected_regions

    selected_model_families = st.multiselect(
        "Select Model Families", options=model_families, default=model_families.tolist()
    )
    if selected_model_families != st.session_state.get("selected_model_families", []):
        st.session_state.selected_model_families = selected_model_families

    if st.button("Apply"):
            st.session_state.apply_filters_clicked = True

def apply_filters():
    logging.info("Applying filters to test data...")
    
    # assume y_test and preds are aligned
    assert len(st.session_state.y_test) == len(st.session_state.preds)
    assert len(st.session_state.test_data) == len(st.session_state.y_test)

    test_data = st.session_state.test_data
    selected_categories = st.session_state.selected_scenario_categories
    selected_regions = st.session_state.selected_regions
    selected_families = st.session_state.selected_model_families

    mask_td = (
        (test_data['Scenario_Category'].isin(selected_categories) if selected_categories else True) &
        (test_data['Region'].isin(selected_regions) if selected_regions else True) &
        (test_data['Model_Family'].isin(selected_families) if selected_families else True)
    )
    
    st.session_state.test_mask = mask_td
    logging.info(f"Test data filter applied: {mask_td.sum()} rows selected.")
    logging.info(f"Selected Categories: {selected_categories}")
    logging.info(f"Selected Regions: {selected_regions}")
    logging.info(f"Selected Model Families: {selected_families}")

    selected_positions = np.where(mask_td)[0]

    assert len(st.session_state.y_test) == len(st.session_state.preds)
    mask_targets = np.zeros(len(st.session_state.y_test), dtype=bool)
    mask_targets[selected_positions] = True
    st.session_state.target_mask = mask_targets

def filter_and_plot():
    logging.info("Filtering test data and plotting results...")

    filtered_y_test = st.session_state.y_test[st.session_state.target_mask]
    filtered_preds = st.session_state.preds[st.session_state.target_mask]
    filtered_test_data = st.session_state.test_data[st.session_state.test_mask].reset_index(drop=True)

    plot_time_series(
        filtered_test_data,
        filtered_y_test,
        filtered_preds,
        st.session_state.targets,
        alpha=0.5,
        linewidth=0.5
    )

def main():
    run_id = "run_02"
    
    if st.session_state.get("logging_initialized", False) is False:
        setup_logging(run_id)
        st.session_state.logging_initialized = True

    st.title("ML IAM Emulation Dashboard")
    session_state = load_session_state(run_id)
    if not session_state:
        st.error("No trained model found. Run train_test.py first.")
        return

    st.session_state.update(session_state)
    make_filters(st.session_state.test_data)
    
    if st.session_state.get("apply_filters_clicked", False):
        apply_filters()
        if st.session_state.target_mask.sum() == 0:
            st.warning("No data selected with the current filters.")
        else:
            filter_and_plot()
        st.session_state.apply_filters_clicked = False

if __name__ == "__main__":
    main()