# Simple task runner for data processing + training

SHELL := /bin/bash
.ONESHELL:

.PHONY: process-data train train-bg stop status dashboard unit-test check-env

# Allow overrides via environment variables (resolved at recipe time under conda)
RAW_DIR ?=
DATA_DIR ?=
RESULTS_DIR ?=

# Conda activation (auto-detect from conda on PATH)
CONDA_SH ?= $(shell conda info --base 2>/dev/null)/etc/profile.d/conda.sh

# The environment to run in. Set CONDA_ENV to empty to use whichever
# environment is already activated, which is what a venv or uv user wants.
CONDA_ENV ?= ml-iam

# Enter environment $(1) and set BIN to the directory its commands live in.
#
# `conda activate` reliably sets CONDA_PREFIX, but it does not always win the
# PATH race against a conda bin that the surrounding shell injected earlier --
# in a VSCode terminal here, `python` after activation is still the base
# interpreter, which has this repo installed and would run the pipeline under
# the wrong library versions without a word.  Addressing each command by path
# instead of by name closes that hole.
define enter_env
	if [ -n "$(1)" ]; then \
		if [ ! -r "$(CONDA_SH)" ]; then \
			echo "ERROR: no conda profile at '$(CONDA_SH)'."; \
			echo "Set CONDA_SH=<conda base>/etc/profile.d/conda.sh, or pass an empty env to use the active one."; \
			exit 1; \
		fi; \
		source "$(CONDA_SH)"; \
		conda activate "$(1)" || { \
			echo "ERROR: conda env '$(1)' not found."; \
			echo "Create it (see README), or override CONDA_ENV / DASHBOARD_ENV on the make command line."; \
			exit 1; \
		}; \
	fi; \
	BIN="$${CONDA_PREFIX:+$$CONDA_PREFIX/bin/}"; \
	PY="$$BIN"python; \
	if [ ! -x "$$PY" ]; then PY="$$(command -v python || true)"; BIN=""; fi; \
	if [ -z "$$PY" ]; then echo "ERROR: no python interpreter found."; exit 1; fi
endef

# Report which interpreter the other targets will use, and what it has.
# The first thing to run when a target behaves as if it were in another env.
check-env:
	@$(call enter_env,$(CONDA_ENV))
	echo "CONDA_ENV     = $(CONDA_ENV)"
	echo "DASHBOARD_ENV = $(DASHBOARD_ENV)"
	"$$PY" scripts/env_report.py

process-data:
	@$(call enter_env,$(CONDA_ENV))
	echo "Using interpreter: $$PY"
	RAW_DIR="$${RAW_DIR:-$$("$$PY" -c 'import configs.paths as c; print(c.RAW_DATA_PATH)')}" ; \
	DATA_DIR="$${DATA_DIR:-$$("$$PY" -c 'import configs.paths as c; print(c.DATA_PATH)')}" ; \
	RESULTS_DIR="$${RESULTS_DIR:-$$("$$PY" -c 'import configs.paths as c; print(c.RESULTS_PATH)')}" ; \
	"$$PY" -m src.data.process_data \
		--raw-dir "$$RAW_DIR" \
		--data-dir "$$DATA_DIR" \
		--results-dir "$$RESULTS_DIR"


# ----------------------
# Unified training entrypoints
# ----------------------

# Run config file (YAML/JSON) used by scripts/train_from_config.py
RUN ?=

# Foreground training (prints run_id to stdout)
train:
	@set -e
	set -o pipefail
	@if [ -z "$(RUN)" ]; then \
		echo "ERROR: RUN is required (e.g. RUN=configs/runs/xgb_example.yaml)"; \
		exit 2; \
	fi
	$(call enter_env,$(CONDA_ENV))
	echo "Using interpreter: $$PY"
	"$$PY" scripts/train_from_config.py --run "$(RUN)"


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


# Stop a background session (training or dashboard)
PID_FILE ?=
stop:
	@if [ -z "$(PID_FILE)" ]; then \
		echo "ERROR: PID_FILE is required (e.g. make stop PID_FILE=logs/train_20260326_110407.pid)"; \
		echo "Active sessions:"; \
		ls -t $(LOG_DIR)/train_*.pid $(LOG_DIR)/dashboard_*.pid 2>/dev/null || echo "  (none)"; \
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


# List active sessions (training + dashboard)
status:
	@echo "=== Active sessions ==="; \
	found=0; \
	for pidfile in $(LOG_DIR)/train_*.pid $(LOG_DIR)/dashboard_*.pid; do \
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
# Unit tests
# ----------------------

# Fast, data-free regression tests (no GPU, no dataset required)
unit-test:
	@$(call enter_env,$(CONDA_ENV))
	"$$PY" -m pytest tests -q


# ----------------------
# Dashboard
# ----------------------

# RUN_ID: pass a specific run (e.g. xgb_76) or just a model type (xgb, lstm, tft)
RUN_ID ?= xgb
# Optional: save individual plots (comma-separated indices, default: 6)
SAVE_PLOTS ?= 6

# The dashboard runs in its own environment so Streamlit's dependencies stay
# out of the training stack.  requirements-dashboard.txt describes how to
# build it; override on the command line if yours is named differently
# (make dashboard DASHBOARD_ENV=<name>).
DASHBOARD_ENV ?= mliam_st

dashboard:
	@mkdir -p "$(LOG_DIR)"
	$(call enter_env,$(DASHBOARD_ENV))
	echo "Using streamlit: $$BIN"streamlit
	if [ ! -x "$$BIN"streamlit ]; then \
		echo "ERROR: no streamlit in env '$(DASHBOARD_ENV)'."; \
		echo "Install it there (see requirements.txt), or point DASHBOARD_ENV at an env that has it."; \
		exit 1; \
	fi
	ts=$$(date +%Y%m%d_%H%M%S); \
	log="$(LOG_DIR)/dashboard_$${ts}.log"; \
	pid="$(LOG_DIR)/dashboard_$${ts}.pid"; \
	setsid nohup env SAVE_INDIVIDUAL_PLOTS=true INDIVIDUAL_PLOT_INDICES="[$(SAVE_PLOTS)]" \
		PYTHONUNBUFFERED=1 \
		"$$BIN"streamlit run scripts/dashboard.py \
		--logger.level=debug \
		--server.runOnSave=false \
		-- --run_id=$(RUN_ID) > "$$log" 2>&1 & \
	echo $$! > "$$pid"; \
	echo "Started dashboard (default run: $(RUN_ID))"; \
	echo "- pidfile: $$pid"; \
	echo "- logfile: $$log"; \
	echo "Tip: http://localhost:8501/?run_id=xgb | lstm | tft"; \
	echo "Stop: make stop PID_FILE=$$pid"
