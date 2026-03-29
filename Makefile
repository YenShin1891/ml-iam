# Simple task runner for data processing + training

SHELL := /bin/bash
.ONESHELL:

.PHONY: process-data train train-bg stop status dashboard

# Allow overrides via environment variables (resolved at recipe time under conda)
RAW_DIR ?=
DATA_DIR ?=
RESULTS_DIR ?=

process-data:
	source "$(CONDA_SH)"
	conda activate "$(CONDA_ENV)"
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

# Conda activation (auto-detect from conda on PATH)
CONDA_SH ?= $(shell conda info --base 2>/dev/null)/etc/profile.d/conda.sh
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
	conda activate "$(CONDA_ENV)"
	python scripts/train_from_config.py --run "$(RUN)"


# Background training via nohup; writes logs + pid under ./logs/
# Uses setsid to create a dedicated process group so that `make stop`
# can cleanly terminate the entire tree (including DDP workers).
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
	setsid nohup $(MAKE) train RUN="$(RUN)" > "$$log" 2>&1 & \
	echo $$! > "$$pid"; \
	echo "Started background training"; \
	echo "- pidfile: $$pid"; \
	echo "- logfile:  $$log"; \
	echo "Tip: tail -f $$log"; \
	echo "Stop: make stop PID_FILE=$$pid"


# Stop a background training session (kills all child processes including DDP workers)
PID_FILE ?=
stop:
	@if [ -z "$(PID_FILE)" ]; then \
		echo "ERROR: PID_FILE is required (e.g. make stop PID_FILE=logs/train_20260326_110407.pid)"; \
		echo "Active sessions:"; \
		ls -t $(LOG_DIR)/train_*.pid 2>/dev/null || echo "  (none)"; \
		exit 2; \
	fi
	@if [ ! -f "$(PID_FILE)" ]; then \
		echo "ERROR: PID file not found: $(PID_FILE)"; \
		exit 1; \
	fi
	@pid=$$(cat "$(PID_FILE)"); \
	echo "Stopping process group for PID $$pid ..."; \
	kill -- -$$pid 2>/dev/null || kill $$pid 2>/dev/null || echo "Process already stopped"; \
	rm -f "$(PID_FILE)"; \
	echo "Done. Verify with: make status"


# List active training sessions
status:
	@echo "=== Active training sessions ==="; \
	found=0; \
	for pidfile in $(LOG_DIR)/train_*.pid; do \
		[ -f "$$pidfile" ] || continue; \
		pid=$$(cat "$$pidfile"); \
		if kill -0 "$$pid" 2>/dev/null && [ -d "/proc/$$pid" ]; then \
			logfile="$${pidfile%.pid}.log"; \
			echo "  PID $$pid ($$pidfile)"; \
			if [ -f "$$logfile" ]; then \
				echo "    Last log: $$(tail -1 "$$logfile")"; \
			fi; \
			found=1; \
		else \
			rm -f "$$pidfile"; \
		fi; \
	done; \
	if [ "$$found" = 0 ]; then echo "  (none)"; fi


# ----------------------
# Dashboard
# ----------------------

# Required: RUN_ID (e.g. xgb_37)
RUN_ID ?=
# Optional: save individual plots (comma-separated indices, default: 6)
SAVE_PLOTS ?= 6

dashboard:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "ERROR: RUN_ID is required (e.g. RUN_ID=xgb_37)"; \
		exit 2; \
	fi
	@mkdir -p "$(LOG_DIR)"
	@ts=$$(date +%Y%m%d_%H%M%S); \
	log="$(LOG_DIR)/dashboard_$${ts}.log"; \
	pid="$(LOG_DIR)/dashboard_$${ts}.pid"; \
	nohup env SAVE_INDIVIDUAL_PLOTS=true INDIVIDUAL_PLOT_INDICES="[$(SAVE_PLOTS)]" \
		streamlit run scripts/dashboard.py \
		--logger.level=info \
		--server.runOnSave=false \
		-- --run_id=$(RUN_ID) > "$$log" 2>&1 & \
	echo $$! > "$$pid"; \
	echo "Started dashboard for run $(RUN_ID)"; \
	echo "- pidfile: $$pid"; \
	echo "- logfile: $$log"; \
	echo "Tip: Open http://localhost:8501"
