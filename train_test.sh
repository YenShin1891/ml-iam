run_id=$(python ./scripts/get_run_id.py)
python ./scripts/train_tft.py --resume=search --run_id=$run_id
python ./scripts/train_tft.py --resume=test --run_id=$run_id