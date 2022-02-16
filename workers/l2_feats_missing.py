from typing import DefaultDict
from typing import Iterable
from collections import defaultdict
from os import getenv

import numpy as np
from messagingclient import MessagingClient
from kvdbclient.base import Entry
from kvdbclient.base import EntryKey
from pychunkedgraph.backend.chunkedgraph import ChunkedGraph
from pychunkedgraph.backend.chunkedgraph_utils import basetypes


def get_l2ids(payload) -> Iterable:
    from pickle import loads

    try:
        return np.frombuffer(payload.data, dtype=basetypes.NODE_ID)
    except (ValueError, TypeError):
        # not numpy array, try pickle
        data = loads(payload.data)
        return np.array(data["new_lvl2_ids"], dtype=basetypes.NODE_ID)


def callback(payload):
    import gc
    import logging
    from cloudvolume import CloudVolume
    from kvdbclient import BigTableClient
    from pcgl2cache.core.attributes import SIZE_NM3
    from pcgl2cache.core.features import run_l2cache
    from pcgl2cache.core.features import write_to_db

    INFO_PRIORITY = 25
    logging.basicConfig(
        level=INFO_PRIORITY,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    graph_id = payload.attributes["table_id"]
    l2ids = get_l2ids(payload)

    l2cache_config = read_l2cache_config()[graph_id]
    if graph_id is not "fly_v31":
        # currently no other graph has cache, ignore
        return
    l2_cache_id = payload.attributes.get("l2_cache_id", l2cache_config["l2cache_id"])

    logging.log(
        INFO_PRIORITY,
        f"Calculating features for {l2ids.size} L2 IDs, graph: {graph_id}, cache: {l2_cache_id}.",
    )

    cv_path = l2cache_config["cv_path"]
    client = BigTableClient(l2_cache_id)
    cg = ChunkedGraph(table_id)
    cv = CloudVolume(
        cv_path, bounded=False, fill_missing=True, progress=False, mip=cg.cv.mip
    )
    for _id in l2ids:
        if cg.get_chunk_layer(_id) != 2:
            continue
        result = run_l2cache(cg, cv, l2id=_id)
        if not result:
            entry = Entry(EntryKey(_id), {SIZE_NM3: np.uint64(0)})
            client.write_entries([entry])
        write_to_db(client, result)
        gc.collect()


c = MessagingClient()
l2cache_update_queue = getenv("L2CACHE_UPDATE_QUEUE", "does-not-exist")
c.consume(l2cache_update_queue, callback)
