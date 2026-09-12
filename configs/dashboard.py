"""Dashboard configuration — default run IDs per model type and the what-if view."""

# When the URL has just ?run_id=xgb (or lstm, tft), resolve to these defaults.
# Update these when you have a new best run for each model.
DEFAULT_RUNS = {
    "xgb": "xgb_76",
    "lstm": "lstm_76",
    "tft": "tft_94",
}

# ── What-if emulator view ─────────────────────────────────────────────────
# Model types the view can run live.  XGB rolls out autoregressively and the
# LSTM predicts fixed windows; neither has an engine here yet.
WHATIF_ENGINES = {"tft"}

# The model is loaded and run on this accelerator.  "cpu" keeps the dashboard
# off the GPUs a training job on the same host may be using.
WHATIF_ACCELERATOR = "cpu"

# Only regions the run emulates well are offered: pooled R2 on the held-out
# test scenarios at or above the threshold, scored on at least this many
# elements (a tiny region's R2 says little either way).
WHATIF_R2_THRESHOLD = 0.95
WHATIF_MIN_SAMPLE_SIZE = 1000
WHATIF_DEFAULT_REGION = "World"

# The trajectory the levers start from: a run of this AR6 category, this
# (Model, Scenario) where the region has it, else the eligible run reporting
# the most inputs.  Every eligible run has at least the window the TFT needs.
# The preferred run is a 2°C (1300 GtCO2 budget) scenario that reports 39 of
# the 42 inputs in every region it covers.
WHATIF_BASELINE_CATEGORY = "C3"
WHATIF_PREFERRED_BASELINE = ("REMIND-MAgPIE 2.0-4.1", "Diff_1300Gt_hybrid_def")

# Levers are bounded by where the training scenarios of that region lie in
# each year: these quantiles of the reported (not imputed) values, from at
# least this many scenarios.  The band is evaluated on the decade grid,
# which every scenario reports, and interpolated in between; the scenarios
# that also report the years in between are a different population, and a
# band read off them zigzags.
WHATIF_BAND_QUANTILES = (0.05, 0.95)
WHATIF_MIN_BAND_COUNT = 5
WHATIF_BAND_YEAR_STEP = 10
# A grid year reported by fewer than this share of the scenarios that report
# the best-covered year (2000, say, which a handful start at) gets no band.
WHATIF_MIN_BAND_FRACTION = 0.5

# Years a lever can be pinned at in the advanced controls; the simple control
# sets the last one and ramps to it.  Anchors before the end of the fixed
# history, or after the trajectory ends, are dropped.
WHATIF_ANCHOR_YEARS = tuple(range(2030, 2101, 10))

# Inputs shown as sliders without opening the "All inputs" section.
WHATIF_KEY_LEVERS = [
    "Population",
    "GDP|MER",
    "GDP|PPP",
    "Price|Carbon",
    "Yield|Cereal",
    "Capital Cost|Electricity|Solar|PV",
    "Capital Cost|Electricity|Wind|Onshore",
    "Capital Cost|Electricity|Nuclear",
    "Capital Cost|Electricity|Coal|w/o CCS",
    "Capital Cost|Electricity|Gas|w/o CCS",
]

# One-click settings: each lever is moved to this position within its band
# at the last year (0 = lower quantile, 1 = upper), ramping from history.
WHATIF_PRESETS = {
    "High carbon price": {"Price|Carbon": 0.95},
    "Cheap solar": {"Capital Cost|Electricity|Solar|PV": 0.05},
    "Low population": {"Population": 0.05},
    "High growth": {"GDP|MER": 0.95, "GDP|PPP": 0.95},
}
