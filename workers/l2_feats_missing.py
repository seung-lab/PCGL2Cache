from typing import DefaultDict
from typing import Iterable
from collections import defaultdict
from os import getenv

import numpy as np
from messagingclient import MessagingClient
from pychunkedgraph.backend.chunkedgraph import ChunkedGraph


def get_batches(cg: ChunkedGraph, l2ids: Iterable) -> DefaultDict:
    chunk_ids = cg.get_chunk_ids_from_node_ids(l2ids)
    chunk_l2id_map = defaultdict(list)
    for k, v in zip(chunk_ids, l2ids):
        chunk_l2id_map[k].append(v)
    return chunk_l2id_map


def callback(payload):
    import logging
    from cloudvolume import CloudVolume
    from kvdbclient import BigTableClient
    from pcgl2cache.core.features import run_l2cache
    from pcgl2cache.core.features import write_to_db

    # l2ids = np.frombuffer(payload.data, dtype=np.uint64)
    # table_id = payload.attributes["table_id"]
    # l2_cache_id = payload.attributes["l2_cache_id"]

    table_id = "fly_v31"
    l2_cache_id = "l2cache_fly_v31_v1"

    import random
    from itertools import product

    cg = ChunkedGraph("fly_v31")
    chunks = list(product(*[range(r) for r in (212, 110, 14)]))
    print(len(chunks))

    l2ids = []
    for i in range(100):
        c = random.choice(chunks)
        rr = cg.range_read_chunk(layer=2, x=c[0], y=c[1], z=c[2])
        l2ids.extend(list(rr.keys()))
        if len(l2ids):
            break
    l2ids = np.array(l2ids[:1], dtype=np.uint64)

    logging.basicConfig(level=logging.INFO)
    logging.info(
        f"Calculating features for {l2ids.size} L2 IDs, graph: {table_id}, cache: {l2_cache_id}."
    )
    cv_path = getenv(
        "CV_GRAPHENE_PATH",
        "graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31",
    )

    client = BigTableClient(l2_cache_id)
    cg = ChunkedGraph(table_id)
    cv = CloudVolume(
        cv_path, bounded=False, fill_missing=True, progress=False, mip=cg.cv.mip
    )
    # chunk_l2ids_map = get_batches(cg, l2ids)
    # for batch in chunk_l2ids_map.values():
    #     result = run_l2cache(cg, cv, l2_ids=batch)
    #     write_to_db(client, result)

    for _id in l2ids:
        result = run_l2cache(cg, cv, l2id=_id)


callback(None)

# c = MessagingClient()
# l2cache_update_queue = getenv("L2CACHE_UPDATE_QUEUE", "test")
# c.consume(l2cache_update_queue, callback)
