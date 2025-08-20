# Directories
DATA_PATH = "/mnt/nas2/kcp-yeeun/data/processed"
RESULTS_PATH = "/mnt/nas2/kcp-yeeun/results"

# Constants
YEAR_RANGE = ('2005', '2100')
MAX_SERIES_LENGTH = 15

DATASET_NAME = 'processed_series_0401.csv'
OUTPUT_VARIABLES = [
    "Emissions|CO2", "Emissions|CH4", "Emissions|N2O",
    "Primary Energy|Coal", "Primary Energy|Gas", "Primary Energy|Oil",
    "Primary Energy|Solar", "Primary Energy|Wind", "Primary Energy|Nuclear"
]
OUTPUT_UNITS = [
    "Mt CO2/yr", "Mt CH4/yr", "Mt N2O/yr", 
    "PJ/yr", "PJ/yr", "PJ/yr", 
    "PJ/yr", "PJ/yr", "PJ/yr"
]
INDEX_COLUMNS = ['Model', 'Scenario', 'Region']
NON_FEATURE_COLUMNS = ['Model', 'Scenario', 'Scenario_Category']
CATEGORICAL_COLUMNS = ['Region', 'Model_Family']