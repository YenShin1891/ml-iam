#!/bin/bash

# XGBoost training and testing script with proper phase separation
# Search phase uses multi-GPU (1 process per visible GPU), train/test/plot phases run in separate commands

set -e
set -o pipefail

source /root/conda/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate xgb2

run_id=$(python ./scripts/get_run_id.py --model xgb)
echo "Starting XGBoost training with run_id: $run_id"

# Allow overriding lag enforcement via LAG_REQUIRED=false
lag_flag="--lag-required"
if [ "${LAG_REQUIRED:-true}" = "false" ]; then
    lag_flag="--no-lag-required"
fi

if [ -z "${DATASET:-}" ]; then
    echo "DATASET not set, using default dataset."
    echo "Phase 1: Running hyperparameter search..."
    python ./scripts/train_xgb.py --resume=search --run_id=$run_id $lag_flag

    echo "Phase 2: Running final training..."
    python ./scripts/train_xgb.py --resume=train --run_id=$run_id $lag_flag

    echo "Phase 3: Running testing..."
    python ./scripts/train_xgb.py --resume=test --run_id=$run_id $lag_flag

    echo "Phase 4: Plotting results..."
    python ./scripts/train_xgb.py --resume=plot --run_id=$run_id $lag_flag
else
    echo "Using dataset override: $DATASET"
    echo "Phase 1: Running hyperparameter search..."
    python ./scripts/train_xgb.py --resume=search --run_id=$run_id --dataset=$DATASET $lag_flag

    echo "Phase 2: Running final training..."
    python ./scripts/train_xgb.py --resume=train --run_id=$run_id --dataset=$DATASET $lag_flag

    echo "Phase 3: Running testing..."
    python ./scripts/train_xgb.py --resume=test --run_id=$run_id --dataset=$DATASET $lag_flag

    echo "Phase 4: Plotting results..."
    python ./scripts/train_xgb.py --resume=plot --run_id=$run_id --dataset=$DATASET $lag_flag
fi

echo "XGBoost training pipeline completed for run_id: $run_id"
