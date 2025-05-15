import dask

dask.config.set({
    'distributed.worker.memory.target': 0.7,
    'distributed.worker.memory.spill': 0.8,
    'distributed.worker.memory.pause': 0.85,
    'distributed.worker.memory.terminate': 0.95,
    'distributed.worker.memory.recent-to-old-time': '30s',
    'distributed.logging.distributed': 'warning',
})