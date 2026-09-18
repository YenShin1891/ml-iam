#!/usr/bin/env python
"""Emulate new scenarios with a trained run.

    python scripts/predict.py --run_id tft_95 --describe
    python scripts/predict.py --run_id tft_95 --input my_scenarios.csv --output emulated.csv

The input is a CSV in the layout of the processed dataset: columns Model,
Scenario, Region, Variable, then one column per year.  Model_Family and
Region_Scale are derived when absent.  ``--describe`` lists the input
Variables, regions and model families the run knows, and whether it needs
target history (XGBoost does; TFT and LSTM do not).

The run must be under RESULTS_PATH (configs/paths.py): train it, or fetch a
published one with ``make fetch-models``.  No training data is needed.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_id", required=True, help="e.g. xgb_85, lstm_89, tft_95")
    parser.add_argument("--input", help="scenario table (CSV)")
    parser.add_argument("--output", help="where to write the predictions (CSV)")
    parser.add_argument("--format", choices=("tidy", "iamc"), default="tidy",
                        help="tidy: a row per (series, Year), a column per target; "
                             "iamc: a row per Variable, a column per year")
    parser.add_argument("--device", default="cpu", help="cpu (default) or gpu; only the TFT uses it")
    parser.add_argument("--describe", action="store_true", help="print what the run expects and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    from src.inference.new_inputs import (
        InputError,
        describe_requirements,
        predict_new_inputs,
        read_scenario_table,
        to_iamc,
    )

    if args.describe:
        print(json.dumps(describe_requirements(args.run_id), indent=2))
        return 0
    if not args.input or not args.output:
        parser.error("--input and --output are required unless --describe is given")

    try:
        table = read_scenario_table(args.input)
        predictions, notes = predict_new_inputs(args.run_id, table, device=args.device)
    except (InputError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    targets = [c for c in predictions.columns if c not in ("Model", "Scenario", "Region", "Year")]
    out = to_iamc(predictions, targets) if args.format == "iamc" else predictions
    out.to_csv(args.output, index=False)
    for note in notes:
        print(f"note: {note}")
    n_series = predictions[["Model", "Scenario", "Region"]].drop_duplicates().shape[0]
    print(f"{args.run_id}: emulated {n_series} series, {len(predictions)} timesteps -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
