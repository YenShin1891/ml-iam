# Simple task runner for data processing + training

SHELL := /bin/bash
.ONESHELL:

.PHONY: process-data train train-bg dashboard

# Allow overrides via environment variables (resolved at recipe time under conda)
RAW_DIR ?=
DATA_DIR ?=
RESULTS_DIR ?=

process-data:
	source "$(CONDA_SH)"
	eval "$$(mamba shell hook --shell bash)"
	mamba activate "$(CONDA_ENV)"
	RAW_DIR="$${RAW_DIR:-$$(python -c 'import configs.paths as c; print(c.RAW_DATA_PATH)')}" ; \
	DATA_DIR="$${DATA_DIR:-$$(python -c 'import configs.paths as c; print(c.DATA_PATH)')}" ; \
	RESULTS_DIR="$${RESULTS_DIR:-$$(python -c 'import configs.paths as c; print(c.RESULTS_PATH)')}" ; \
	python -m src.data.process_data \
		--raw-dir "$$RAW_DIR" \
		--data-dir "$$DATA_DIR" \
		--results-dir "$$RESULTS_DIR"


# ----------------------
# Unified training entrypoints
# ----------------------

# Run config file (YAML/JSON) used by scripts/train_from_config.py
RUN ?=

# Conda/mamba activation (mirrors existing train_test_*.sh scripts)
CONDA_SH ?= /root/conda/etc/profile.d/conda.sh
CONDA_ENV ?= ml-iam

# Foreground training (prints run_id to stdout)
train:
	set -e
	set -o pipefail
	@if [ -z "$(RUN)" ]; then \
		echo "ERROR: RUN is required (e.g. RUN=configs/runs/xgb_example.yaml)"; \
		exit 2; \
	fi
	source "$(CONDA_SH)"
	eval "$$(mamba shell hook --shell bash)"
	mamba activate "$(CONDA_ENV)"
	python scripts/train_from_config.py --run "$(RUN)"


# Background training via nohup; writes logs + pid under ./logs/
LOG_DIR ?= logs
train-bg:
	set -e
	set -o pipefail
	@if [ -z "$(RUN)" ]; then \
		echo "ERROR: RUN is required (e.g. RUN=configs/runs/xgb_example.yaml)"; \
		exit 2; \
	fi
	@mkdir -p "$(LOG_DIR)"
	@ts=$$(date +%Y%m%d_%H%M%S); \
	log="$(LOG_DIR)/train_$${ts}.log"; \
	pid="$(LOG_DIR)/train_$${ts}.pid"; \
	nohup $(MAKE) train RUN="$(RUN)" > "$$log" 2>&1 & \
	echo $$! > "$$pid"; \
	echo "Started background training"; \
	echo "- pidfile: $$pid"; \
	echo "- logfile:  $$log"; \
	echo "Tip: tail -f $$log"


# ----------------------
# Dashboard
# ----------------------

# Required: RUN_ID (e.g. xgb_37)
RUN_ID ?=
# Optional: save individual plots (comma-separated indices)
SAVE_PLOTS ?=

dashboard:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "ERROR: RUN_ID is required (e.g. RUN_ID=xgb_37)"; \
		exit 2; \
	fi
	@export SAVE_INDIVIDUAL_PLOTS=false; \
	export INDIVIDUAL_PLOT_INDICES='[]'; \
	if [ -n "$(SAVE_PLOTS)" ]; then \
		export SAVE_INDIVIDUAL_PLOTS=true; \
		export INDIVIDUAL_PLOT_INDICES="[$(SAVE_PLOTS)]"; \
	fi; \
	nohup streamlit run scripts/dashboard.py \
		--logger.level=info \
		--server.runOnSave=false \
		-- --run_id=$(RUN_ID) &
	@echo "Dashboard started for run $(RUN_ID)"
	@echo "Tip: Open http://localhost:8501"
