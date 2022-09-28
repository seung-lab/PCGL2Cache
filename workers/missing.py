from os import getenv

import numpy as np
from messagingclient import MessagingClient


def callback(payload):
    import logging
    from pcgl2cache.utils import read_l2cache_config
    from .common import calculate_features

    INFO_PRIORITY = 25
    logging.basicConfig(
        level=INFO_PRIORITY,
        format="%(asctime)s %(message)s",
        datefmt="%m-%d-%Y %I:%M:%S %p",
    )

    graph_id = payload.attributes["table_id"]
    l2ids = np.frombuffer(payload.data, dtype=np.uint64)
    try:
        l2cache_config = read_l2cache_config()[graph_id]
    except KeyError:
        logging.error(f"Config for {graph_id} not found.")
        # ignore datasets without l2cache
        return

    l2cache_id = payload.attributes.get("l2_cache_id", l2cache_config["l2cache_id"])
    calculate_features(l2ids, l2cache_id, l2cache_config["cv_path"])
    logging.log(
        INFO_PRIORITY,
        f"Calculated features for {l2ids.size} L2 IDs {l2ids[:5]}..., graph: {graph_id}, cache: {l2cache_id}.",
    )


c = MessagingClient()
l2cache_update_queue = getenv("L2CACHE_UPDATE_QUEUE", "does-not-exist")
c.consume(l2cache_update_queue, callback)
