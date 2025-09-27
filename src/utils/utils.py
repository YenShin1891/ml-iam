import inspect
import json
import logging
import os
import pickle
from typing import Optional
from datetime import datetime, timezone, timedelta
from dask.distributed import Client

from configs.paths import RESULTS_PATH

# constants
CHECKPOINT_FILE_NAME = "session_state.pkl"

# for logging
class KSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        kst = timezone(timedelta(hours=9))  # KST is UTC+9
        record_time = datetime.fromtimestamp(record.created, tz=kst)
        return record_time.strftime(datefmt or '%Y-%m-%d %H:%M:%S')
    

def _make_formatter(fmt: str = '%(asctime)s - %(levelname)s - %(message)s') -> logging.Formatter:
    """Return a KST-based formatter with the standard format."""
    return KSTFormatter(fmt)


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

    log_dir = os.path.join(RESULTS_PATH, run_id, "logs")
    os.makedirs(log_dir, exist_ok=True)

    handlers = [
        _make_stream_handler(logging.INFO),
        _make_file_handler(os.path.join(log_dir, log_file), logging.INFO),
    ]

    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logging.info("Logging is set up for %s.", run_id)


# for model checkpoints (legacy simple pickle helpers retained below)


def get_next_run_id():
    """
    Generate the next run_id by checking existing run directories in the results folder.
    """
    os.makedirs(RESULTS_PATH, exist_ok=True)

    existing_runs = [
        d for d in os.listdir(RESULTS_PATH)
        if os.path.isdir(os.path.join(RESULTS_PATH, d)) and d.startswith("run_")
    ]

    # Extract numeric parts of run IDs and find the next available number
    run_numbers = [
        int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()
    ]
    next_run_number = max(run_numbers, default=0) + 1
    return f"run_{next_run_number:02d}"


# for model checkpoints
def save_session_state(session_state, run_id, checkpoint_file_name=CHECKPOINT_FILE_NAME):
    """
    Save the session state to a file under the specified run directory.
    """
    run_dir = os.path.join(RESULTS_PATH, run_id, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    file_path = os.path.join(run_dir, checkpoint_file_name)
    with open(file_path, "wb") as f:
        pickle.dump(session_state, f)
    logging.info("Session state saved to %s.", file_path)

class DowngradeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map new numpy modules to old ones
        if module == 'numpy._core._exceptions':
            module = 'numpy.core._exceptions'
        elif module == 'numpy._core._dtype':
            module = 'numpy.core._dtype'
        elif module == 'numpy._core._internal':
            module = 'numpy.core._internal'
        elif module == 'numpy._core._methods':
            module = 'numpy.core._methods'
        elif module.startswith('numpy._core'):
            # Generic mapping for other _core modules
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)
    

def load_session_state(run_id, checkpoint_file_name=CHECKPOINT_FILE_NAME):
    file_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", checkpoint_file_name)
    try:
        with open(file_path, "rb") as f:
            return DowngradeUnpickler(f).load()
        
    except FileNotFoundError:
        logging.error("No saved session state found at %s.", file_path)
        return {}
    
def load_model(run_id):
    run_dir = os.path.join(RESULTS_PATH, run_id, "checkpoints")
    file_path = os.path.join(run_dir, "final_best.json")
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(file_path)
        return model
    except Exception as e:
        logging.error("Error loading model: %s", str(e))
        return None

def load_best_params(session_state):
    """
    Load best_params from best_params.json file in current directory.
    """
    param_path = os.path.join(RESULTS_PATH, "best_params.json")
    with open(param_path, "r") as f:
        best_params = json.load(f)
    
    # Convert parameter names to match XGBoost training expectations
    if 'n_estimators' in best_params:
        best_params['num_boost_round'] = best_params.pop('n_estimators')
    if 'learning_rate' in best_params:
        best_params['eta'] = best_params.pop('learning_rate')
    
    session_state["best_params"] = best_params
    logging.info("Loaded best_params from best_params.json: %s", best_params)
    
    return session_state

# dask
def create_dask_client():
    """
    Create a Dask client for distributed computing.
    """
    return Client(
        n_workers=1,
        threads_per_worker=2,
        memory_limit='4GB',
        silence_logs=logging.WARNING,
        dashboard_address=None,
        local_directory='/tmp/dask-worker-space'  
    )
