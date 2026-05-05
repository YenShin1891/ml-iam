"""Dashboard configuration — default run IDs per model type."""

# When the URL has just ?run_id=xgb (or lstm, tft), resolve to these defaults.
# Update these when you have a new best run for each model.
DEFAULT_RUNS = {
    "xgb": "xgb_76",
    "lstm": "lstm_76",
    "tft": "tft_86",
}
