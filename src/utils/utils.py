import inspect
import logging
import os
import pickle
from datetime import datetime, timezone, timedelta
from dask.distributed import Client

from configs.config import RESULTS_PATH

# constants
CHECKPOINT_FILE_NAME = "session_state.pkl"

# for logging
class KSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        kst = timezone(timedelta(hours=9))  # KST is UTC+9
        record_time = datetime.fromtimestamp(record.created, tz=kst)
        return record_time.strftime(datefmt or '%Y-%m-%d %H:%M:%S')
    

def setup_logging(run_id, log_file=None):
    """
    Set up logging with a log file under the specified run directory.
    """
    if log_file is None:
        caller_filename = inspect.stack()[1].filename
        log_file = os.path.basename(caller_filename).split('.')[0] + ".log"

    log_dir = os.path.join(RESULTS_PATH, run_id, "logs")
    os.makedirs(log_dir, exist_ok=True)

    handlers = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(os.path.join(log_dir, log_file)))

    formatter = KSTFormatter('%(asctime)s - %(levelname)s - %(message)s')

    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers
    )
    logging.info("Logging is set up for %s.", run_id)


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

def load_session_state(run_id, checkpoint_file_name=CHECKPOINT_FILE_NAME):
    file_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", checkpoint_file_name)
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.error("No saved session state found at %s.", file_path)
        return {}


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
