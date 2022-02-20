from itertools import product
from datetime import datetime
from typing import Sequence
from typing import Optional

import numpy as np

from ..utils import chunk_id_str
from ..manager import IngestionManager
from pychunkedgraph.backend.chunkedgraph import ChunkedGraph


def _post_task_completion(imanager: IngestionManager, layer: int, coords: np.ndarray):
    chunk_str = "_".join(map(str, coords))
    # remove from queued hash and put in completed hash
    imanager.redis.sadd(f"{layer}c", chunk_str)
    return


def enqueue_atomic_tasks(
    imanager: IngestionManager, cv_path: str, timestamp: Optional[datetime] = None
):
    from time import sleep
    from rq import Queue as RQueue
    from ..utils import chunked

    imanager.redis.flushdb()
    bbox = np.array(imanager.cg.cv.bounds.to_list())
    dataset_size = bbox[3:] - bbox[:3]
    atomic_chunk_bounds = np.ceil(dataset_size / imanager.cg.chunk_size).astype(np.int)
    chunk_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
    np.random.shuffle(chunk_coords)

    if imanager.config.TEST_RUN:
        x, y, z = np.array(atomic_chunk_bounds) // 2
        f = lambda r1, r2, r3: np.array(np.meshgrid(r1, r2, r3), dtype=int).T.reshape(
            -1, 3
        )
        chunk_coords = f((x, x + 1), (y, y + 1), (z, z + 1))

    print(f"Jobs count: {len(chunk_coords)}")
    chunked_jobs = chunked(chunk_coords, 10000)

    for batch in chunked_jobs:
        q = imanager.get_task_queue(imanager.config.CLUSTER.L2CACHE_Q_NAME)
        # for optimal use of redis memory wait if queue limit is reached
        if len(q) > imanager.config.CLUSTER.L2CACHE_Q_LIMIT:
            print(f"Sleeping {imanager.config.CLUSTER.L2CACHE_Q_INTERVAL}s...")
            sleep(imanager.config.CLUSTER.L2CACHE_Q_INTERVAL)

        job_datas = []
        for chunk in batch:
            job_datas.append(
                RQueue.prepare_data(
                    _ingest_chunk,
                    args=(
                        imanager.serialize_info(pickled=True),
                        cv_path,
                        chunk,
                        timestamp,
                    ),
                    timeout="5m",
                    result_ttl=0,
                    job_id=chunk_id_str(2, chunk),
                )
            )

        q.enqueue_many(job_datas)


def _ingest_chunk(
    im_info: str,
    cv_path: str,
    chunk_coord: Sequence[int],
    timestamp: datetime,
):
    from cloudvolume import CloudVolume
    from kvdbclient import BigTableClient
    from ...core.features import run_l2cache
    from ...core.features import write_to_db

    imanager = IngestionManager.from_pickle(im_info)
    cg = ChunkedGraph(imanager.cg.table_id)
    cv = CloudVolume(
        cv_path, bounded=False, fill_missing=True, progress=False, mip=cg.cv.mip
    )

    chunk_coord = np.array(list(chunk_coord), dtype=np.int)
    r = run_l2cache(cv, cg, chunk_coord, timestamp)
    write_to_db(BigTableClient(imanager.cache_id), r)
    _post_task_completion(imanager, 2, chunk_coord)
