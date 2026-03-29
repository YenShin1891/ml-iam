import inspect
import logging
import os
from datetime import datetime
from typing import Optional

from configs.paths import RESULTS_PATH


def get_run_root(run_id: str) -> str:
    """Return the root directory for a run.

    Expected run_id format: "{model}_{nn}" (e.g., "xgb_01", "lstm_03", "tft_12").
    Results are stored under: RESULTS_PATH/{model}/{run_id}
    """
    model_type = run_id.split("_", 1)[0]
    return os.path.join(RESULTS_PATH, model_type, run_id)

class LocalFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        record_time = datetime.fromtimestamp(record.created).astimezone()
        return record_time.strftime(datefmt or '%Y-%m-%d %H:%M:%S')


def _make_formatter(fmt: str = '%(asctime)s - %(levelname)s - %(message)s') -> logging.Formatter:
    return LocalFormatter(fmt)


def _make_stream_handler(level: int = logging.INFO, fmt: Optional[str] = None) -> logging.Handler:
    """Create a StreamHandler with KST formatting."""
    h = logging.StreamHandler()
    if fmt is None:
        fmt = '%(asctime)s - %(levelname)s - %(message)s'
    h.setFormatter(_make_formatter(fmt))
    h.setLevel(level)
    return h


def _make_file_handler(file_path: str, level: int = logging.INFO, fmt: Optional[str] = None) -> logging.Handler:
    """Create a FileHandler with KST formatting."""
    h = logging.FileHandler(file_path)
    if fmt is None:
        fmt = '%(asctime)s - %(levelname)s - %(message)s'
    h.setFormatter(_make_formatter(fmt))
    h.setLevel(level)
    return h


def setup_console_logging(level: int = logging.INFO, logger_name: Optional[str] = None) -> logging.Logger:
    """
    Configure a logger to log to console only using the same KST format as training.

    Args:
        level: Logging level (default INFO)
        logger_name: Optional named logger; if None, configures the root logger.

    Returns:
        The configured logger instance.
    """
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    # Avoid duplicate handlers if called multiple times
    logger.handlers.clear()
    logger.setLevel(level)
    logger.addHandler(_make_stream_handler(level))
    # When using a named logger, avoid propagating to root to prevent duplicate prints
    if logger_name:
        logger.propagate = False
    return logger


def setup_logging(run_id, log_file=None):
    """
    Set up logging with a log file under the specified run directory.
    """
    if log_file is None:
        caller_filename = inspect.stack()[1].filename
        log_file = os.path.basename(caller_filename).split('.')[0] + ".log"

    log_dir = os.path.join(get_run_root(run_id), "logs")
    os.makedirs(log_dir, exist_ok=True)

    handlers = [
        _make_stream_handler(logging.INFO),
        _make_file_handler(os.path.join(log_dir, log_file), logging.INFO),
    ]

    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logging.info("Logging is set up for %s.", run_id)


def get_next_run_id(model_type: str) -> str:
    """
    Generate the next run_id for a given model.

    This function is concurrency-safe: it reserves the run directory atomically to
    avoid collisions when multiple jobs start at the same time (e.g. via nohup).

    Returns a model-prefixed id like "xgb_01".
    """
    model_results_dir = os.path.join(RESULTS_PATH, model_type)
    os.makedirs(model_results_dir, exist_ok=True)

    # Start from current max+1 to avoid O(N) retries in the common case.
    existing_runs = [
        d
        for d in os.listdir(model_results_dir)
        if os.path.isdir(os.path.join(model_results_dir, d)) and d.startswith(f"{model_type}_")
    ]
    run_numbers = []
    for d in existing_runs:
        try:
            suffix = d.split("_", 1)[1]
        except Exception:
            continue
        if suffix.isdigit():
            try:
                run_numbers.append(int(suffix))
            except Exception:
                continue
    candidate = max(run_numbers, default=0) + 1

    # Atomically reserve a unique run directory.
    # If another process races us, mkdir will fail with FileExistsError and we retry.
    while True:
        run_id = f"{model_type}_{candidate:02d}"
        run_root = os.path.join(model_results_dir, run_id)
        try:
            os.makedirs(run_root, exist_ok=False)
            return run_id
        except FileExistsError:
            candidate += 1


def load_model(run_id):
    run_dir = os.path.join(get_run_root(run_id), "checkpoints")
    file_path = os.path.join(run_dir, "final_best.json")
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(file_path)
        return model
    except Exception as e:
        logging.error("Error loading model: %s", str(e))
        return None
