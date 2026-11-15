"""The runs the paper reports: the single place that names them.

Anything that should follow the paper (the dashboard's shortcuts, the
defaults of the figure scripts) reads the run ids from here.  What is
published for download is a separate list, metadata/published_models.json:
it names the archives that exist on Zenodo and changes only with an upload.
"""

# One run per model, in the order the figures show them.
PAPER_RUNS = {
    "xgb": "xgb_90",
    "lstm": "lstm_89",
    "tft": "tft_106",
}

# The ridge regression the three are compared against.
PAPER_BASELINE_RUN = "linear_04"

# A repeat of the XGBoost run with identical settings and metrics, on eight
# GPUs: the source of the XGBoost timings only.
PAPER_TIMING_RUNS = {"xgb": "xgb_91"}

# Panel titles for the runs above.
MODEL_NAMES = {"xgb": "XGBoost", "lstm": "LSTM", "tft": "TFT"}
