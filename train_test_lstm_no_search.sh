#!/bin/bash

# LSTM training and testing script with proper phase separation
# Search phase uses multi-device, train/test phases use single device

run_id=$(python ./scripts/get_run_id.py)
echo "Starting LSTM training with run_id: $run_id"



if [ -z "$DATASET" ]; then
    echo "DATASET not set, using default dataset."
	echo "Skipping Phase 1 (no_search script)"
	echo "Phase 2: Running final training..."
	python ./scripts/train_lstm.py --skip_search --resume=train --run_id=$run_id

	echo "Phase 3: Running testing..."
	python ./scripts/train_lstm.py --resume=test --run_id=$run_id

	echo "Phase 4: Plotting results..."
	python ./scripts/train_lstm.py --resume=plot --run_id=$run_id
else
    echo "Using dataset override: $DATASET"
	echo "Skipping Phase 1 (no_search script)"
	echo "Phase 2: Running final training..."
	python ./scripts/train_lstm.py --skip_search --resume=train --run_id=$run_id --dataset=$DATASET

	echo "Phase 3: Running testing..."
	python ./scripts/train_lstm.py --resume=test --run_id=$run_id --dataset=$DATASET

	echo "Phase 4: Plotting results..."
	python ./scripts/train_lstm.py --resume=plot --run_id=$run_id --dataset=$DATASET
fi

echo "LSTM training pipeline completed for run_id: $run_id"