import numpy as np
from messagingclient import MessagingClient


def callback(payload):
    from pcgl2cache.core.calc_l2_feats import run_l2cache

    l2ids = np.frombuffer(payload.data, dtype=np.uint64)
    l2ids = np.array([175137943013294114, 175137943013294118], dtype=np.uint64)
    table_id = payload.attributes["table_id"]
    l2_cache_id = payload.attributes["l2_cache_id"]
    print(table_id, l2_cache_id, l2ids)

    cv_path = "graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31"
    r = run_l2cache(table_id, cv_path, l2_ids=l2ids)
    print(r)


c = MessagingClient()
c.consume("test", callback)
