import inspect
import logging
import os
import pickle
from datetime import datetime, timezone, timedelta

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(PARENT_DIR), "results")

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

    log_dir = os.path.join(RESULTS_DIR, run_id, "logs")
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
def save_session_state(session_state, run_id):
    """
    Save the session state to a file under the specified run directory.
    """
    file_name = "session_state.pkl"  # Fixed name for the latest session state
    run_dir = os.path.join(RESULTS_DIR, run_id, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    file_path = os.path.join(run_dir, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(session_state, f)
    logging.info("Session state saved to %s.", file_path)

def load_session_state(file_name):
    file_path = os.path.join(RESULTS_DIR, "checkpoints", file_name)
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.error("No saved session state found at %s.", file_path)
        return {}


def get_next_run_id(results_dir="results"):
    """
    Generate the next run_id by checking existing run directories in the results folder.
    """
    os.makedirs(results_dir, exist_ok=True)

    existing_runs = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("run_")
    ]

    # Extract numeric parts of run IDs and find the next available number
    run_numbers = [
        int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()
    ]
    next_run_number = max(run_numbers, default=0) + 1
    return f"run_{next_run_number:02d}"