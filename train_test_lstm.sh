#!/bin/bash

# LSTM training and testing script with proper phase separation
# Search phase uses multi-device, train/test phases use single device

run_id=$(python ./scripts/get_run_id.py)
echo "Starting LSTM training with run_id: $run_id"

# Phase 1: Hyperparameter search (multi-device)
echo "Phase 1: Running hyperparameter search..."
python ./scripts/train_lstm.py --resume=search --run_id=$run_id

# Phase 2: Final training (multi device)
echo "Phase 2: Running final training..."
python ./scripts/train_lstm.py --resume=train --run_id=$run_id

# Phase 3: Testing (single device, clean environment)
echo "Phase 3: Running testing..."
python ./scripts/train_lstm.py --resume=test --run_id=$run_id

# Phase 4: Plotting results
echo "Phase 4: Plotting results..."
python ./scripts/train_lstm.py --resume=plot --run_id=$run_id

echo "LSTM training pipeline completed for run_id: $run_id"