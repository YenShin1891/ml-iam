run_id=$(python ./scripts/get_run_id.py)
python ./scripts/train_tft.py --resume=search --run_id=$run_id --dataset="pipeline-min10100-comp0.4-out=9vars-apply-base-year-2025-09-26"
python ./scripts/train_tft.py --resume=test --run_id=$run_id --dataset="pipeline-min10100-comp0.4-out=9vars-apply-base-year-2025-09-26"