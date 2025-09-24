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

# Units for each output variable (single source of truth)
# Keys must exactly match entries in OUTPUT_VARIABLES
UNITS_BY_OUTPUT = {
    "Primary Energy|Coal": "PJ/yr",
    "Primary Energy|Gas": "PJ/yr",
    "Primary Energy|Oil": "PJ/yr",
    "Primary Energy|Solar": "PJ/yr",
    "Primary Energy|Wind": "PJ/yr",
    "Primary Energy|Nuclear": "PJ/yr",
    "Emissions|CO2": "Mt CO2/yr",
    "Emissions|CH4": "Mt CH4/yr",
    "Emissions|N2O": "Mt N2O/yr",
}

# Convenience: units list aligned to OUTPUT_VARIABLES order
OUTPUT_UNITS = [UNITS_BY_OUTPUT[var] for var in OUTPUT_VARIABLES]

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
NAME_PREFIX = "pipeline"
INCLUDE_DATE = True
DATE_FMT = "%Y-%m-%d"
# Tags include dynamic count of output variables
# Pre-defined tags: include-intermediate, apply-base-year
TAGS = [f"out={len(OUTPUT_VARIABLES)}vars", "exclude-year", "apply-base-year"]
SAVE_ANALYSIS = True

# Optional data structure hints (used by TFT & plotting)
INDEX_COLUMNS = ['Model', 'Scenario', 'Region']
NON_FEATURE_COLUMNS = ['Model', 'Scenario', 'Scenario_Category', 'Year']
CATEGORICAL_COLUMNS = ['Region', 'Model_Family']

# Feature engineering knobs for downstream (kept here for single stop)
MAX_SERIES_LENGTH = 10
N_LAG_FEATURES = 3
MAX_YEAR = 2100  # Upper inclusive cutoff for usable year columns

# Default dataset name (backward compatibility for older scripts)
DEFAULT_DATASET = 'processed_series_0401.csv'
