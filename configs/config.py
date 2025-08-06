# Directories
DATA_PATH = "/mnt/nas2/kcp-yeeun/data/processed"
RESULTS_PATH = "/mnt/nas3/kcp-yeeun/results"

# Constants
YEAR_RANGE = ('2005', '2100')
SPLIT_POINT = 2  # '2005, '2010' is given as context
LAGGING = 2  # TODO: Not sure if every instance has data starting 2005

DATASET_NAME = 'processed_series_0401.csv'
OUTPUT_VARIABLES = [
    "Emissions|CO2", "Emissions|CH4", "Emissions|N2O",
    "Primary Energy|Coal", "Primary Energy|Gas", "Primary Energy|Oil",
    "Primary Energy|Solar", "Primary Energy|Wind", "Primary Energy|Nuclear"
]
INDEX_COLUMNS = ['Model', 'Scenario', 'Region']
NON_FEATURE_COLUMNS = ['Model', 'Scenario', 'Scenario_Category']
STATIC_FEATURE_COLUMNS = ['Model_Family', 'Region', 'target']
