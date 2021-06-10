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


def enqueue_atomic_tasks(imanager: IngestionManager):
    from time import sleep

    imanager.redis.flushdb()
    chunk_coords = _get_test_chunks(imanager.cg.meta)

    if not imanager.config.TEST_RUN:
        atomic_chunk_bounds = imanager.cg_meta.layer_chunk_bounds[2]
        chunk_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
        np.random.shuffle(chunk_coords)

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
            args=(imanager.serialize_info(pickled=True), chunk_coord),
        )


def _ingest_atomic_chunk(im_info: str, coord: Sequence[int]):
    imanager = IngestionManager.from_pickle(im_info)
    coord = np.array(list(coord), dtype=np.int)

    cv_path = "graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31"
    run_l2cache_preproc(imanager.cg.table_id, cv_path)

    _post_task_completion(imanager, 2, coord)


def _get_test_chunks(meta: ChunkedGraphMeta):
    """
    Returns chunks that lie at the center of the dataset
    """
    f = lambda r1, r2, r3: np.array(np.meshgrid(r1, r2, r3), dtype=int).T.reshape(-1, 3)
    x, y, z = np.array(meta.layer_chunk_bounds[2]) // 2
    return f((x, x + 1), (y, y + 1), (z, z + 1))
