"""Rebuild metadata/scenario_category.csv from the AR6 metadata workbook.

AR6 assigns its climate category per (Model, Scenario), not per scenario
name: the same name, say SSP2-Baseline, is run by many models and lands in a
different category under each.  The table therefore carries the Model column
and process_data merges on both keys.

Usage (needs openpyxl, which the training environment does not ship):

    python scripts/build_scenario_category.py
"""
from pathlib import Path

import pandas as pd

from configs.paths import RAW_DATA_PATH

WORKBOOK = Path(RAW_DATA_PATH) / "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
SHEET = "meta"  # every scenario; the vetted sheet drops the unassessed ones
OUTPUT = Path(__file__).resolve().parents[1] / "metadata" / "scenario_category.csv"


def main() -> None:
    meta = pd.read_excel(WORKBOOK, sheet_name=SHEET, dtype=str)
    table = (
        meta[["Model", "Scenario", "Category", "Category_name"]]
        .rename(columns={"Category": "Scenario_Category"})
        .sort_values(["Model", "Scenario"])
        .reset_index(drop=True)
    )
    duplicated = table.duplicated(["Model", "Scenario"])
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} (Model, Scenario) pairs appear more than once in {WORKBOOK}")
    table.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(table)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
