import inspect
import json
import logging
import os
import pickle
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype

from configs.paths import RESULTS_PATH
from configs.data import REGION_CODE_TO_LABEL

# constants
CHECKPOINT_FILE_NAME = "session_state.pkl"


def get_run_root(run_id: str) -> str:
    """Return the root directory for a run.

    Expected run_id format: "{model}_{nn}" (e.g., "xgb_01", "lstm_03", "tft_12").
    Results are stored under: RESULTS_PATH/{model}/{run_id}
    """
    model_type = run_id.split("_", 1)[0]
    return os.path.join(RESULTS_PATH, model_type, run_id)

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

    log_dir = os.path.join(get_run_root(run_id), "logs")
    os.makedirs(log_dir, exist_ok=True)

    handlers = [
        _make_stream_handler(logging.INFO),
        _make_file_handler(os.path.join(log_dir, log_file), logging.INFO),
    ]

    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logging.info("Logging is set up for %s.", run_id)


# for model checkpoints (legacy simple pickle helpers retained below)


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


# for model checkpoints
def save_session_state(session_state, run_id, checkpoint_file_name=CHECKPOINT_FILE_NAME):
    """
    Save the session state to a file under the specified run directory.
    """
    run_dir = os.path.join(get_run_root(run_id), "checkpoints")
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
    file_path = os.path.join(get_run_root(run_id), "checkpoints", checkpoint_file_name)
    try:
        with open(file_path, "rb") as f:
            session_state = DowngradeUnpickler(f).load()
        _decode_saved_categoricals(session_state)
        return session_state
        
    except FileNotFoundError:
        logging.debug("No saved session state found at %s; starting fresh.", file_path)
        return {}


def _decode_saved_categoricals(session_state: dict) -> None:
    """Restore categorical label columns (e.g., Region) that may have been encoded as integers."""
    if not isinstance(session_state, dict):
        return

    def _decode_region_column(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            return
        if 'Region' not in df.columns:
            return
        region_series = df['Region']
        if not is_numeric_dtype(region_series):
            return
        try:
            codes = region_series.astype('Int64')
        except Exception:
            return

        def _map_code(value):
            if pd.isna(value):
                return None
            try:
                code_int = int(value)
            except (TypeError, ValueError):
                return None
            if code_int < 0:
                return None
            return REGION_CODE_TO_LABEL.get(code_int)

        decoded = codes.map(_map_code)
        df.loc[:, 'Region'] = decoded.astype('object')

    for key, value in list(session_state.items()):
        if isinstance(value, pd.DataFrame):
            _decode_region_column(value)
    
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

def masked_mse(y_true, y_pred):
    import tensorflow as tf
    # Ensure both y_true and y_pred are float32
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    # 마스크: NaN이 아닌 값만 True
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, dtype=tf.float32)

    # NaN을 0으로 대체해서 계산
    y_true = tf.where(mask == 1.0, y_true, tf.zeros_like(y_true))
    y_pred = tf.where(mask == 1.0, y_pred, tf.zeros_like(y_pred))

    squared_error = tf.square(y_true - y_pred)

    # 평균은 마스크된 값으로만 계산
    masked_loss = tf.reduce_sum(squared_error * mask) / tf.reduce_sum(mask)
    return masked_loss
