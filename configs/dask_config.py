import dask
import logging

dask.config.set({
    'distributed.worker.memory.target': 0.7,
    'distributed.worker.memory.spill': 0.8,
    'distributed.worker.memory.pause': 0.85,
    'distributed.worker.memory.terminate': 0.95,
    'distributed.worker.memory.recent-to-old-time': '30s',
    'distributed.logging.distributed': 'warning',
})

CLIENT_CONFIGS = {
    'n_workers': 1,
    'threads_per_worker': 2,
    'memory_limit': '4GB',
    'silence_logs': logging.WARNING,
    'dashboard_address': None,
    'local_directory': '/tmp/dask-worker-space'
}
