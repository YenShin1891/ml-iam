from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class XGBRuntimeConfig:
    """Runtime and cluster settings for XGBoost with optional Dask.

    Mirrors TFT's config style (dataclasses, explicit build/apply) while keeping
    compatibility with existing trainer code that uses a CLIENT_CONFIGS dict.
    """

    # Dask distributed config to set at runtime (no side effects at import time)
    dask_distributed_config: Dict[str, Any] = field(
        default_factory=lambda: {
            'distributed.worker.memory.target': 0.7,
            'distributed.worker.memory.spill': 0.8,
            'distributed.worker.memory.pause': 0.85,
            'distributed.worker.memory.terminate': 0.95,
            'distributed.worker.memory.recent-to-old-time': '30s',
            'distributed.logging.distributed': 'warning',
            'distributed.worker.daemon': False,
            'distributed.comm.compression': 'lz4',
            'distributed.worker.use-file-locking': False,
        }
    )

    # Dask LocalCluster/Client kwargs
    client_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            'n_workers': 1,
            'threads_per_worker': 1,
            'memory_limit': '16GB',
            'silence_logs': 30,  # logging.WARNING
            'dashboard_address': None,
            'local_directory': '/tmp/dask-worker-space',
            'death_timeout': 60,
        }
    )

    def apply(self) -> None:
        """Apply Dask runtime settings if Dask is available."""
        try:
            import dask  # type: ignore
            dask.config.set(self.dask_distributed_config)
        except Exception:
            # Allow non-Dask runtimes to proceed
            pass


@dataclass
class XGBTrainerConfig:
    """Trainer defaults for XGB, mirroring TFTTrainerConfig style."""

    # Base xgboost params used by trainer's get_xgb_params
    tree_method: str = 'hist'
    device: str = 'cuda'
    eval_metric: str = 'rmse'
    verbosity: int = 0
    max_bin: int = 256

    # Training loop controls
    early_stopping_rounds: int = 15
    n_folds: int = 5
    search_iter_n_per_stage: int = 15
    use_dask: bool = True

    # GPU affinity (for manual round-robin in trainer)
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


__all__ = [
    'XGBRuntimeConfig',
    'XGBTrainerConfig',
]
