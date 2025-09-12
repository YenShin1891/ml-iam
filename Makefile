# Simple task runner for data processing

.PHONY: process-data

# Allow overrides via environment variables
RAW_DIR ?= $(shell python -c 'import configs.paths as c; print(c.RAW_DATA_PATH)')
DATA_DIR ?= $(shell python -c 'import configs.paths as c; print(c.DATA_PATH)')
RESULTS_DIR ?= $(shell python -c 'import configs.paths as c; print(c.RESULTS_PATH)')

process-data:
	python -m src.data.process_data \
		--raw-dir "$(RAW_DIR)" \
		--data-dir "$(DATA_DIR)" \
		--results-dir "$(RESULTS_DIR)"
