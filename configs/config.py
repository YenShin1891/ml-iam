# Directories
DATA_PATH = "/mnt/nas3/kcp-yeeun/data/processed"
RESULTS_PATH = "/mnt/nas3/kcp-yeeun/results"

# Constants
YEAR_STARTS_AT = 6
DATASET_NAME = 'processed_series_0401.csv'

# Model configuration
YEAR_RANGE = ('2020', '2100')
MAX_SERIES_LENGTH = 10
N_LAG_FEATURES = 3  # Number of lagged features to create (1, 2, 3, etc.)

# Target variables
OUTPUT_VARIABLES = [
    "Primary Energy|Coal", 
    "Primary Energy|Gas", 
    "Primary Energy|Oil",
    "Primary Energy|Solar", 
    "Primary Energy|Wind", 
    "Primary Energy|Nuclear",
    "Emissions|CO2", 
    "Emissions|CH4", 
    "Emissions|N2O",
]

# Data structure
INDEX_COLUMNS = ['Model', 'Scenario', 'Region']
NON_FEATURE_COLUMNS = ['Model', 'Scenario', 'Scenario_Category']
CATEGORICAL_COLUMNS = ['Region', 'Model_Family']