from itertools import product
from typing import Sequence

import numpy as np

from ..utils import chunk_id_str
from ..manager import IngestionManager
from pychunkedgraph.backend import ChunkedGraphMeta


def _post_task_completion(imanager: IngestionManager, layer: int, coords: np.ndarray):
    chunk_str = "_".join(map(str, coords))
    # remove from queued hash and put in completed hash
    imanager.redis.hdel(f"{layer}q", chunk_str)
    imanager.redis.hset(f"{layer}c", chunk_str, "")
    return


def enqueue_atomic_tasks(imanager: IngestionManager, cv_path: str):
    from time import sleep

    imanager.redis.flushdb()

    bbox = np.array(imanager.cg.cv.bounds.to_list())
    dataset_size = bbox[3:] - bbox[:3]
    atomic_chunk_bounds = np.ceil(dataset_size / imanager.cg.chunk_size).astype(np.int)
    chunk_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
    np.random.shuffle(chunk_coords)

    if imanager.config.TEST_RUN:
        mid = len(chunk_coords) / 2
        chunk_coords = chunk_coords[mid : mid + 10]

    for chunk_coord in chunk_coords:
        atomic_queue = imanager.get_task_queue(imanager.config.CLUSTER.ATOMIC_Q_NAME)
        # for optimal use of redis memory wait if queue limit is reached
        if len(atomic_queue) > imanager.config.CLUSTER.ATOMIC_Q_LIMIT:
            print(f"Sleeping {imanager.config.CLUSTER.ATOMIC_Q_INTERVAL}s...")
            sleep(imanager.config.CLUSTER.ATOMIC_Q_INTERVAL)
        atomic_queue.enqueue(
            _ingest_atomic_chunk,
            job_id=chunk_id_str(2, chunk_coord),
            job_timeout="2m",
            result_ttl=0,
            args=(imanager.serialize_info(pickled=True), cv_path, chunk_coord),
        )


def _ingest_atomic_chunk(im_info: str, cv_path: str, chunk_coord: Sequence[int]):
    from ...proc.calc_l2_feats import run_l2cache

    imanager = IngestionManager.from_pickle(im_info)
    chunk_coord = np.array(list(chunk_coord), dtype=np.int)
    run_l2cache(imanager.cg.table_id, cv_path, chunk_coord)
    _post_task_completion(imanager, 2, chunk_coord)
