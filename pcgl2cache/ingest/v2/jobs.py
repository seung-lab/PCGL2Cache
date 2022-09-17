from itertools import product
from datetime import datetime
from typing import Tuple
from typing import Sequence
from typing import Optional
from os import environ

import numpy as np

from ..utils import chunk_id_str
from ..manager import IngestionManager


def _post_task_completion(imanager: IngestionManager, layer: int, coords: np.ndarray):
    chunk_str = "_".join(map(str, coords))
    # remove from queued hash and put in completed hash
    imanager.redis.sadd(f"{layer}c", chunk_str)
    return


def randomize_grid_points(X: int, Y: int, Z: int) -> Tuple[int, int, int]:
    indices = np.arange(X * Y * Z)
    np.random.shuffle(indices)
    for index in indices:
        yield np.unravel_index(index, (X, Y, Z))


def enqueue_atomic_tasks(
    imanager: IngestionManager, cv_path: str, timestamp: Optional[datetime] = None
):
    from time import sleep
    from rq import Queue as RQueue
    from ..utils import chunked

    atomic_chunk_bounds = imanager.cg.meta.layer_chunk_bounds[2]
    chunk_coords = randomize_grid_points(*atomic_chunk_bounds)

    if imanager.config.TEST_RUN:
        x, y, z = np.array(atomic_chunk_bounds) // 2
        f = lambda r1, r2, r3: np.array(np.meshgrid(r1, r2, r3), dtype=int).T.reshape(
            -1, 3
        )
        chunk_coords = f((x, x + 1), (y, y + 1), (z, z + 1))
        print(f"Test jobs count: {len(chunk_coords)}")

    print(f"Total jobs count: {imanager.cg.meta.layer_chunk_counts[0]}")
    batch_size = int(environ.get("L2JOB_BATCH_SIZE", 1000))

    job_datas = []
    for chunk_coord in chunk_coords:
        q = imanager.get_task_queue(imanager.config.CLUSTER.L2CACHE_Q_NAME)
        # buffer for optimal use of redis memory
        if len(q) > imanager.config.CLUSTER.L2CACHE_Q_LIMIT:
            print(f"Sleeping {imanager.config.CLUSTER.L2CACHE_Q_INTERVAL}s...")
            sleep(imanager.config.CLUSTER.L2CACHE_Q_INTERVAL)

        x, y, z = chunk_coord
        chunk_str = f"{x}_{y}_{z}"
        if imanager.redis.sismember("2c", chunk_str):
            # already done, skip
            continue
        job_datas.append(
            RQueue.prepare_data(
                _ingest_chunk,
                args=(
                    imanager.serialize_info(pickled=True),
                    cv_path,
                    chunk_coord,
                    timestamp,
                ),
                timeout=environ.get("JOB_TIMEOUT", "5m"),
                result_ttl=0,
                job_id=chunk_id_str(2, chunk_coord),
            )
        )
        if len(job_datas) % batch_size == 0:
            q.enqueue_many(job_datas)
            job_datas = []
    q.enqueue_many(job_datas)


def _ingest_chunk(
    im_info: str,
    cv_path: str,
    chunk_coord: Sequence[int],
    timestamp: datetime,
):
    from cloudvolume import CloudVolume
    from pychunkedgraph.graph import ChunkedGraph
    from kvdbclient import BigTableClient
    from kvdbclient import get_default_client_info
    from ...core.features import run_l2cache
    from ...core.features import write_to_db

    imanager = IngestionManager.from_pickle(im_info)
    cg = ChunkedGraph(graph_id=imanager.graph_id)
    cv = CloudVolume(
        cv_path,
        bounded=False,
        fill_missing=True,
        progress=False,
        mip=cg.meta.cv.mip,
    )

    chunk_coord = np.array(chunk_coord, dtype=np.int)
    r = run_l2cache(
        cv,
        cg=cg,
        chunk_coord=chunk_coord,
        timestamp=timestamp,
    )

    config = get_default_client_info().CONFIG
    print(f"L2ID count: {len(r.get('l2id', []))}")

    write_to_db(BigTableClient(imanager.cache_id, config=config), r)
    _post_task_completion(imanager, 2, chunk_coord)
