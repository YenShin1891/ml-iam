## IAM Emulation with Machine Learning

This repository provides a reproducible pipeline for emulating Integrated Assessment Model (IAM) scenario time‑series using multiple ML approaches:

* Gradient Boosted Trees (XGBoost)
* Temporal Fusion Transformer (TFT, PyTorch Lightning)
* Long Short-Term Memory (LSTM, PyTorch Lightning)

It standardizes data ingestion from the IPCC AR6 Scenario Explorer, feature engineering, model search / training, evaluation, and exploratory visualization via a Streamlit dashboard.

---
## Contents
1. Features
2. Quick Start
3. Environment & Installation
4. Data & Licensing (AR6)
5. Configuration
6. Data Processing
7. Training Pipelines (XGB / TFT / LSTM)
8. Visualization & Explainability
9. Dashboard
10. Project Layout
11. Recommended Citation
12. License
13. FAQ

---
## 1. Features
* Unified AR6 scenario preprocessing & feature engineering.
* Modular model families: XGBoost / TFT / LSTM.
* Resumable pipelines with per‑`run_id` persisted state.
* Autoregressive evaluation + SHAP & visualization utilities.
* Streamlit dashboard (exploratory, WIP).

---
## 2. Quick Start
```bash
git clone <this-repo-url>
cd ml-iam
python -m venv .venv && source .venv/bin/activate  # or use conda
pip install -r requirements.txt

# (Optional) Export environment variables to override defaults
export RAW_DATA_PATH="/path/to/raw" \
	DATA_PATH="/path/to/processed" \
	RESULTS_PATH="/path/to/results"

# Process data (writes processed & intermediate artifacts)
make process-data

# Train XGBoost full pipeline (search -> train -> test -> plot)
python scripts/train_xgb.py

# Train TFT full pipeline
python scripts/train_tft.py

# Train LSTM full pipeline
python scripts/train_lstm.py

# Launch dashboard (foreground)
streamlit run scripts/dashboard.py
```

---
## 3. Environment & Installation
Prerequisites:
* Python 3.9
* (Optional GPU) CUDA 11.6 compatible stack for TFT acceleration
* Open ports (default Streamlit 8501)

Install dependencies:
```bash
pip install -r requirements.txt
```

If using GPUs, confirm PyTorch + CUDA alignment (adjust `pip`/`pip3` index URL if needed).

---
## 4. Data & Licensing (AR6)
The pipeline expects AR6 Scenario Explorer CSVs (v1.1) referenced in `configs/data.py` (`RAW_FILENAMES`). Obtain them from the IPCC / IIASA AR6 Scenario Explorer.

License & Terms of Use: The AR6 data are subject to the license described at:
https://data.ene.iiasa.ac.at/ar6/#/license

Before using or distributing processed outputs, review the above license and attribution guidance. Include appropriate citation(s) in any derivative work. This repository does not redistribute the raw AR6 CSVs.

Place raw files under a directory you configure (see `configs/paths.py` or environment overrides below). Example:
```
/path/to/raw/AR6_Scenarios_Database_ISO3_v1.1.csv
... other region variants ...
```

---
## 5. Configuration
Central configuration modules:
* `configs/data.py` – variable selection (`OUTPUT_VARIABLES`), filtering thresholds, naming/version knobs.
* `configs/paths.py` – path constants (generic placeholders recommended). Replace hard‑coded defaults or export environment variables.
* `configs/models/*.py` – model‑specific hyperparameter search spaces (TFT / XGB).

Override paths without editing code by exporting (evaluated early through Python in the Makefile):
```bash
export RAW_DATA_PATH="/path/to/raw"
export DATA_PATH="/path/to/processed"
export RESULTS_PATH="/path/to/results"
```

Key data knobs (from `configs/data.py`):
* `OUTPUT_VARIABLES` – target columns (energy & emissions series).
* `MIN_COUNT`, `COMPLETENESS_RATIO` – filtering heuristics.
* `MAX_SERIES_LENGTH`, `N_LAG_FEATURES`, `YEAR_RANGE` – feature engineering scope.

---
## 6. Data Processing
Data ingestion & processing is orchestrated via the Makefile target:
```bash
make process-data
```
This runs `python -m src.data.process_data` with directories resolved from `configs.paths` (or environment overrides). Outputs include processed wide/long series and optional analysis artifacts (if `SAVE_ANALYSIS=True`).

---
## 7. Training Pipelines
Each model family has a dedicated driver script in `scripts/`.

### XGBoost (`scripts/train_xgb.py`)
Full pipeline (preprocess -> search -> train -> test -> plot):
```bash
python scripts/train_xgb.py
```
Options:
* `--skip_search` – skip hyperparameter search and directly train using stored/previous best params.
* `--resume {search|train|test|plot} --run_id <id>` – resume from a saved session.
* `--dataset <name>` – specify a processed dataset version.
* `--note "description"` – attach a note stored in session state.

Example background run:
```bash
nohup python scripts/train_xgb.py --note "baseline xgb" > xgb.out 2>&1 &
```

### Temporal Fusion Transformer (`scripts/train_tft.py`)
Runs sequential phases automatically when invoked without arguments:
```bash
python scripts/train_tft.py
```
Resume at a later step:
```bash
python scripts/train_tft.py --resume train --run_id 2024_09_15_001
```

### LSTM (`scripts/train_lstm.py`)
PyTorch Lightning implementation with sequence modeling capabilities:
```bash
python scripts/train_lstm.py
```
Resume at a later step:
```bash
python scripts/train_lstm.py --resume train --run_id 2024_09_15_001
```

---
## 8. Visualization & Explainability
The visualization layer covers three categories:
1. Diagnostic scatter (predicted vs actual across targets)
2. Trajectory comparison (model vs historical / IAM reference)
3. SHAP explainability (For both trees & neural networks)

Directory structure (`src/visualization/`):
```
trajectories.py  # 1 & 2: trajectory panels, scatter diagnostics, metadata helpers
shap_xgb.py      # 3: XGBoost SHAP value computation + summary plots
shap_nn.py       # 3: LSTM, TFT temporal SHAP, heatmap, timestep importance
helpers.py       # Shared: subplot grids, feature name formatting, render helpers
__init__.py      # Public re-exports
```

## 9. Dashboard
The Streamlit dashboard (`scripts/dashboard.py`) provides exploratory visualization (WIP). Launch in foreground:
```bash
streamlit run scripts/dashboard.py
```
Background with basic logging:
```bash
nohup streamlit run scripts/dashboard.py --logger.level=info --server.runOnSave=false > dashboard.out 2>&1 &
```

---
## 10. Project Layout
```
├── configs/
│   ├── data.py            # Data selection & feature engineering knobs
│   ├── paths.py           # Centralized path placeholders
│   └── models/            # Model-specific search configs (tft/xgb)
├── scripts/
│   ├── train_xgb.py       # XGBoost pipeline driver
│   ├── train_tft.py       # TFT pipeline driver
│   ├── train_lstm.py      # LSTM pipeline driver
│   └── dashboard.py       # Streamlit dashboard (exploration)
├── src/
│   ├── data/              # Preprocessing, feature engineering, dataset builders
│   ├── trainers/          # Training loops, search routines, evaluation helpers
│   ├── visualization/     # Plotting & SHAP (trajectories, xgb shap, nn shap, helpers)
│   ├── utils/             # General utilities
├── lightning_logs/        # PyTorch Lightning run artifacts (TFT)
├── metadata/              # Auxiliary classification / scenario metadata
├── requirements.txt       # Python dependencies
├── Makefile               # Data processing convenience target
└── README.md
```

---
## 11. Recommended Citation
If you use this pipeline or derivative artifacts in academic or policy work, cite:
* The IPCC AR6 Scenario Explorer per its official citation guidance.
* (Placeholder for forthcoming paper)

---
## 12. License
This repository's code is released under the existing LICENSE file. AR6 data are subject to their own license; you must obtain and use them in compliance with: https://data.ene.iiasa.ac.at/ar6/#/license

---
## 13. FAQ
**Q:** How do I add a new target variable?  
**A:** Append it to `OUTPUT_VARIABLES` in `configs/data.py`, re-run `make process-data`, then retrain models.

---
Feel free to open issues for clarifications or enhancements.

