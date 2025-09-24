#!/bin/bash

# TFT training and testing script with proper phase separation
# Search phase uses multi-device, train/test phases use single device

run_id=$(python ./scripts/get_run_id.py)
echo "Starting TFT training with run_id: $run_id"



if [ -z "$DATASET" ]; then
	echo "DATASET not set, using default dataset."
	echo "Phase 1: Running hyperparameter search..."
	python ./scripts/train_tft.py --resume=search --run_id=$run_id

	echo "Phase 2: Running final training..."
	python ./scripts/train_tft.py --resume=train --run_id=$run_id

	echo "Phase 3: Running testing..."
	python ./scripts/train_tft.py --resume=test --run_id=$run_id

	echo "Phase 4: Plotting results..."
	python ./scripts/train_tft.py --resume=plot --run_id=$run_id
else
	echo "Using dataset override: $DATASET"
	echo "Phase 1: Running hyperparameter search..."
	python ./scripts/train_tft.py --resume=search --run_id=$run_id --dataset=$DATASET

	echo "Phase 2: Running final training..."
	python ./scripts/train_tft.py --resume=train --run_id=$run_id --dataset=$DATASET

	echo "Phase 3: Running testing..."
	python ./scripts/train_tft.py --resume=test --run_id=$run_id --dataset=$DATASET

	echo "Phase 4: Plotting results..."
	python ./scripts/train_tft.py --resume=plot --run_id=$run_id --dataset=$DATASET
fi

echo "TFT training pipeline completed for run_id: $run_id"