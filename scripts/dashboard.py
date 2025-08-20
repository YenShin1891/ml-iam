import streamlit as st
import numpy as np
import logging

from src.utils.plotting import plot_time_series, get_saved_plots_metadata
from src.utils.utils import setup_logging, load_session_state
import datetime

# Apply global styling for wider sidebar
st.markdown("""
<style>
    .css-1d391kg, [data-testid="stSidebar"] {
        width: 25rem !important;
        min-width: 25rem !important;
    }
    .css-1d391kg > div {
        width: 25rem !important;
        min-width: 25rem !important;
    }
</style>
""", unsafe_allow_html=True)

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
    import os
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

    if st.button("Make New Plot"):
            st.session_state.apply_filters_clicked = True

def apply_filters():
    logging.info("Applying filters to test data...")
    # XGBoost case: has y_test directly
    if hasattr(st.session_state, 'y_test') and st.session_state.y_test is not None:
        y_test = st.session_state.y_test
        preds = st.session_state.preds
        test_data = st.session_state.test_data
    # TFT case: extract y_test from test_data using targets, handle horizon subset
    elif hasattr(st.session_state, 'test_data') and hasattr(st.session_state, 'targets'):
        preds = st.session_state.preds
        targets = st.session_state.targets
        
        # Use horizon subset if available (TFT predictions are on forecast horizon)
        horizon_df = st.session_state.get('horizon_df')
        horizon_y_true = st.session_state.get('horizon_y_true')
        
        if horizon_df is not None and horizon_y_true is not None:
            test_data = horizon_df
            y_test = horizon_y_true
        else:
            st.error("TFT horizon data not found in session state.")
            return
    else:
        st.error("Required data not found in session state. Please ensure the model has been trained.")
        return

    # Build filter mask
    mask_conditions = []
    if 'Scenario_Category' in test_data.columns:
        mask_conditions.append(test_data['Scenario_Category'].isin(st.session_state.selected_scenario_categories))
    if 'Region' in test_data.columns:
        mask_conditions.append(test_data['Region'].isin(st.session_state.selected_regions))
    if 'Model_Family' in test_data.columns:
        mask_conditions.append(test_data['Model_Family'].isin(st.session_state.selected_model_families))
    
    # Combine conditions
    mask_td = mask_conditions[0] if mask_conditions else pd.Series([True] * len(test_data))
    for condition in mask_conditions[1:]:
        mask_td = mask_td & condition
    
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

    # Prepare filter metadata
    filter_metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'scenario_categories': st.session_state.selected_scenario_categories,
        'regions': st.session_state.selected_regions,
        'model_families': st.session_state.selected_model_families,
        'num_data_points': len(filtered_test_data),
        'targets': st.session_state.targets
    }

    # Get environment variables for individual plot saving
    # If you want to save individual plots, use the following command:
    # nohup bash -c "export SAVE_INDIVIDUAL_PLOTS=true && export INDIVIDUAL_PLOT_INDICES='[0]' && streamlit run scripts/dashboard.py --logger.level=info --server.runOnSave=false" &
    import os
    save_individual = os.getenv('SAVE_INDIVIDUAL_PLOTS', 'false').lower() == 'true'
    logging.info(f"DEBUG: save_individual = {save_individual}")
    
    if save_individual:
        individual_indices_str = os.getenv('INDIVIDUAL_PLOT_INDICES', '[0]')
        try:
            individual_indices = eval(individual_indices_str)
            if not isinstance(individual_indices, list):
                individual_indices = [0]
        except:
            individual_indices = [0]
        logging.info(f"DEBUG: individual_indices = {individual_indices}")
    else:
        individual_indices = []
    
    plot_time_series(
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
            st.image(img, caption="Time Series Plot", use_container_width=True)

def setup_session_and_logging(run_id):
    """Initialize logging and load session state."""
    if st.session_state.get("logging_initialized", False) is False:
        setup_logging(run_id)
        st.session_state.logging_initialized = True

    session_state = load_session_state(run_id)
    if not session_state:
        st.error("No trained model found. Run train_test.py first.")
        return None
    
    st.session_state.update(session_state)
    return session_state

def handle_filtering_and_plotting(run_id):
    """Handle the filter application and plotting logic."""
    if st.session_state.get("apply_filters_clicked", False):
        apply_filters()
        if st.session_state.target_mask.sum() == 0:
            st.warning("No data selected with the current filters.")
        else:
            filter_and_plot(run_id)
        st.session_state.apply_filters_clicked = False

def main():
    run_id = "run_39"
    
    # Initialize session and logging
    session_state = setup_session_and_logging(run_id)
    if not session_state:
        return

    # Main dashboard UI
    st.title("ML IAM Emulation Dashboard")
    make_filters(st.session_state.test_data)
    
    # Handle filtering and plotting
    handle_filtering_and_plotting(run_id)
    
    # Recent plots section
    display_recent_plots_sidebar(run_id)
    
    # Display selected plot section
    display_selected_plot()

if __name__ == "__main__":
    main()