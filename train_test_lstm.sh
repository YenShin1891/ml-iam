#!/bin/bash

# LSTM training and testing script with proper phase separation
# Search phase uses multi-device, train/test phases use single device

source /root/conda/etc/profile.d/conda.sh
conda activate xgb2

run_id=$(python ./scripts/get_run_id.py)
echo "Starting LSTM training with run_id: $run_id"



# Allow overriding lag enforcement via LAG_REQUIRED=false
lag_flag="--lag-required"
if [ "${LAG_REQUIRED:-true}" = "false" ]; then
    lag_flag="--no-lag-required"
fi


if [ -z "$DATASET" ]; then
    echo "DATASET not set, using default dataset."
	echo "Phase 1: Running hyperparameter search..."
	python ./scripts/train_lstm.py --resume=search --run_id=$run_id $lag_flag

	echo "Phase 2: Running final training..."
	python ./scripts/train_lstm.py --resume=train --run_id=$run_id $lag_flag

	echo "Phase 3: Running testing..."
	python ./scripts/train_lstm.py --resume=test --run_id=$run_id $lag_flag

	echo "Phase 4: Plotting results..."
	python ./scripts/train_lstm.py --resume=plot --run_id=$run_id $lag_flag
else
    echo "Using dataset override: $DATASET"
	echo "Phase 1: Running hyperparameter search..."
	python ./scripts/train_lstm.py --resume=search --run_id=$run_id --dataset=$DATASET $lag_flag

	echo "Phase 2: Running final training..."
	python ./scripts/train_lstm.py --resume=train --run_id=$run_id --dataset=$DATASET $lag_flag

	echo "Phase 3: Running testing..."
	python ./scripts/train_lstm.py --resume=test --run_id=$run_id --dataset=$DATASET $lag_flag

	echo "Phase 4: Plotting results..."
	python ./scripts/train_lstm.py --resume=plot --run_id=$run_id --dataset=$DATASET $lag_flag
fi

echo "LSTM training pipeline completed for run_id: $run_id"