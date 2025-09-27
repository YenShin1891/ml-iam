#!/usr/bin/env python3
"""
Validation script to check if the dataset has proper 5-year intervals.

This script can be run independently to validate data quality without
affecting the performance of the main data loading pipeline.

Usage:
    python scripts/validate_intervals.py [--dataset DATASET_VERSION] [--fast]
"""

import argparse
import logging
import os
import sys
import time
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.paths import DATA_PATH
from configs.data import DEFAULT_DATASET, MAX_YEAR
from src.data.preprocess import _validate_5year_intervals, _validate_5year_intervals_fast


def load_melted_data(dataset_version=None):
    """Load and melt the dataset for validation."""
    # Determine dataset path
    if dataset_version:
        dataset_path = os.path.join(DATA_PATH, dataset_version, "processed_series.csv")
    else:
        dataset_path = os.path.join(DATA_PATH, DEFAULT_DATASET)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset: {dataset_path}")
    processed_series = pd.read_csv(dataset_path)
    print(f"Loaded dataset with shape: {processed_series.shape}")

    # Identify year and non-year columns
    all_cols = list(processed_series.columns)
    year_cols = [c for c in all_cols if str(c).isdigit()]
    try:
        cutoff = int(MAX_YEAR)
        year_cols = [c for c in year_cols if int(c) <= cutoff]
    except Exception:
        print(f"Warning: MAX_YEAR ({MAX_YEAR}) invalid; using all year columns")

    non_year_cols = [c for c in all_cols if c not in year_cols]

    print(f"Found {len(year_cols)} year columns: {sorted(year_cols)}")
    print(f"Found {len(non_year_cols)} non-year columns: {non_year_cols}")

    # Melt to long format
    print("Melting dataset to long format...")
    year_melted = processed_series.melt(
        id_vars=non_year_cols, value_vars=year_cols, var_name='Year', value_name='value'
    )
    year_melted['Year'] = year_melted['Year'].astype(int)

    print(f"Melted dataset shape: {year_melted.shape}")
    return year_melted


def validate_intervals(melted_df, use_fast=True):
    """Validate 5-year intervals using specified method."""
    group_cols = ['Model', 'Model_Family', 'Scenario', 'Scenario_Category', 'Region']
    num_groups = len(melted_df[group_cols].drop_duplicates())

    print(f"\nValidating 5-year intervals for {num_groups} groups...")
    print(f"Using {'fast vectorized' if use_fast else 'groupby'} method...")

    start_time = time.time()

    try:
        if use_fast:
            _validate_5year_intervals_fast(melted_df)
        else:
            _validate_5year_intervals(melted_df)

        validation_time = time.time() - start_time
        print(f"✓ VALIDATION PASSED in {validation_time:.3f}s")
        print(f"✓ All {num_groups} groups have proper 5-year intervals")
        return True

    except ValueError as e:
        validation_time = time.time() - start_time
        print(f"✗ VALIDATION FAILED in {validation_time:.3f}s")
        print(f"✗ Error: {e}")
        return False
    except Exception as e:
        validation_time = time.time() - start_time
        print(f"✗ UNEXPECTED ERROR in {validation_time:.3f}s")
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_year_patterns(melted_df):
    """Analyze year patterns in the dataset."""
    group_cols = ['Model', 'Model_Family', 'Scenario', 'Scenario_Category', 'Region']

    print("\nAnalyzing year patterns...")

    # Get unique years per group
    group_years = melted_df[group_cols + ['Year']].drop_duplicates()
    year_counts = group_years.groupby(group_cols)['Year'].count()

    print(f"Years per group - Min: {year_counts.min()}, Max: {year_counts.max()}, Mean: {year_counts.mean():.1f}")

    # Show some example groups and their years
    print("\nExample groups and their years:")
    for i, (group_name, group_data) in enumerate(group_years.groupby(group_cols)):
        if i >= 5:  # Show first 5 groups
            break
        years = sorted(group_data['Year'].unique())
        print(f"  {group_name}: {years}")

    # Overall year range
    all_years = sorted(melted_df['Year'].unique())
    print(f"\nOverall year range: {all_years[0]} to {all_years[-1]}")
    print(f"All years: {all_years}")


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(description="Validate 5-year intervals in dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset version to validate (subdirectory under data/)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast vectorized validation (default: True)"
    )
    parser.add_argument(
        "--groupby",
        action="store_true",
        help="Use groupby validation method instead of fast method"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze year patterns in addition to validation"
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        # Load data
        print("=" * 60)
        print("5-YEAR INTERVAL VALIDATION")
        print("=" * 60)

        melted_df = load_melted_data(args.dataset)

        # Analyze patterns if requested
        if args.analyze:
            analyze_year_patterns(melted_df)

        # Validate intervals
        use_fast = not args.groupby  # Default to fast unless --groupby specified
        success = validate_intervals(melted_df, use_fast=use_fast)

        print("\n" + "=" * 60)
        if success:
            print("✓ VALIDATION SUCCESSFUL - Dataset has proper 5-year intervals")
            sys.exit(0)
        else:
            print("✗ VALIDATION FAILED - Dataset does not have proper 5-year intervals")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ SCRIPT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()