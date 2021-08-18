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
    from kvdbclient import BigTableClient
    from pcgl2cache.core.calc_l2_feats import run_l2cache
    from pcgl2cache.core.calc_l2_feats import write_to_db

    l2ids = np.frombuffer(payload.data, dtype=np.uint64)
    table_id = payload.attributes["table_id"]
    l2_cache_id = payload.attributes["l2_cache_id"]

    logging.basicConfig(level=logging.INFO)
    logging.info(
        f"Calculating features for {l2ids.size} L2 IDs, graph: {table_id}, cache: {l2_cache_id}."
    )
    cv_path = getenv(
        "CV_GRAPHENE_PATH",
        "graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31",
    )

    cg = ChunkedGraph(table_id)
    chunk_l2ids_map = get_batches(cg, l2ids)
    ret_dicts = []
    for batch in chunk_l2ids_map.values():
        ret_dicts.append(run_l2cache(cg, cv_path, l2_ids=batch))

    com_dict = defaultdict(list)
    for ret_dict in ret_dicts:
        for k in ret_dict:
            com_dict[k].extend(ret_dict[k])
    write_to_db(BigTableClient(l2_cache_id), com_dict)


c = MessagingClient()
l2cache_update_queue = getenv("L2CACHE_UPDATE_QUEUE", "test")
c.consume(l2cache_update_queue, callback)
