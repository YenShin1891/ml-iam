"""
Data processing configuration: single source of truth for pipeline knobs.
"""

# Column index (0-based) where the first year column begins in raw AR6 CSVs
YEAR_STARTS_AT = 5

# Variable selection targets (single source of truth)
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

# Raw input filenames (AR6 v1.1)
RAW_FILENAMES = [
    "AR6_Scenarios_Database_ISO3_v1.1.csv",
    "AR6_Scenarios_Database_R6_regions_v1.1.csv",
    "AR6_Scenarios_Database_R5_regions_v1.1.csv",
    "AR6_Scenarios_Database_R10_regions_v1.1.csv",
    "AR6_Scenarios_Database_World_v1.1.csv",
]

# Selection and filtering
MIN_COUNT = 10100
COMPLETENESS_RATIO = 0.4

# Versioning / naming
NAME_PREFIX = "processed_series"
INCLUDE_DATE = True
DATE_FMT = "%Y-%m-%d"
TAGS = ["tgt=9vars", "unit=normalized"]
SAVE_ANALYSIS = True

# Optional data structure hints (used by TFT & plotting)
INDEX_COLUMNS = ['Model', 'Scenario', 'Region']
NON_FEATURE_COLUMNS = ['Model', 'Scenario', 'Scenario_Category']
CATEGORICAL_COLUMNS = ['Region', 'Model_Family']

# Feature engineering knobs for downstream (kept here for single stop)
MAX_SERIES_LENGTH = 10
N_LAG_FEATURES = 3
YEAR_RANGE = ('2020', '2100')

# Legacy dataset name (backward compatibility for older scripts)
DATASET_NAME = 'processed_series_0401.csv'
