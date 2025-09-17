import argparse
import logging
import sys

from src.trainers.tft_trainer import predict_tft
from src.utils.utils import load_session_state


def main():
    parser = argparse.ArgumentParser(description="Check if PF provides original-scale predictions.")
    parser.add_argument("--run_id", required=True, help="Run ID with final checkpoint and dataset template")
    args = parser.parse_args()

    try:
        import pytorch_forecasting as pf  # type: ignore
        pf_ver = getattr(pf, "__version__", "unknown")
    except Exception:
        pf_ver = "unavailable"
    try:
        import lightning as L  # type: ignore
        li_ver = getattr(L, "__version__", "unknown")
    except Exception:
        li_ver = "unavailable"
    try:
        import torch
        torch_ver = getattr(torch, "__version__", "unknown")
    except Exception:
        torch_ver = "unavailable"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(f"Environment -> torch={torch_ver} lightning={li_ver} pytorch-forecasting={pf_ver}")

    session_state = load_session_state(args.run_id)
    try:
        _ = predict_tft(session_state, args.run_id)
        logging.info("Result: SUCCESS (Prediction.prediction present and used)")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Result: FAILURE ({e})")
        sys.exit(1)


if __name__ == "__main__":
    main()
