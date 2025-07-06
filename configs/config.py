# Directories
DATA_PATH = "/mnt/nas3/kcp-yeeun/data/processed"
RESULTS_PATH = "/mnt/nas3/kcp-yeeun/results"

# Constants
YEAR_STARTS_AT = 6
DATASET_NAME = 'processed_series_0401.csv'
OUTPUT_VARIABLES = [
    "Emissions|CO2", "Emissions|CH4", "Emissions|N2O",
    "Primary Energy|Coal", "Primary Energy|Gas", "Primary Energy|Oil",
    "Primary Energy|Solar", "Primary Energy|Wind", "Primary Energy|Nuclear"
]
INDEX_COLUMNS = ['Model', 'Scenario', 'Region']
NON_FEATURE_COLUMNS = ['Model', 'Scenario', 'Scenario_Category']
