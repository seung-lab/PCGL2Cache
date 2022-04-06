from typing import Iterable
from os import environ

import numpy as np
from cloudvolume import CloudVolume


def get_chunkedgraph(graph_id):
    if environ.get("CHUNKEDGRAPH_VERSION", "1") == "1":
        from pychunkedgraph.backend.chunkedgraph import ChunkedGraph

        return ChunkedGraph(graph_id)
    from pychunkedgraph.graph import ChunkedGraph

    return ChunkedGraph(graph_id=graph_id)


def read_l2cache_config() -> dict:
    """
    Example yaml file:
    ```
    fly_v31:
      cv_path: "graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31"
      l2cache_id: "l2cache_fly_v31_v2"
    minnie3_v1:
      cv_path: "graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v1"
      l2cache_id: "l2cache_minnie3_v1_v1"
    ```
    """
    import yaml

    try:
        yml_path = environ["GRAPH_L2CACHE_CONFIG_PATH"]
    except KeyError:
        return {}
    with open(yml_path, "r") as stream:
        return yaml.safe_load(stream)


def get_chunk_coordinates(cv: CloudVolume, ids: Iterable) -> Iterable:
    if not len(ids):
        return np.array([])
    ids = np.array(ids)
    layer = cv.get_chunk_layer(ids[0])
    bits_per_dim = cv.meta.spatial_bit_count(layer)

    x_offset = 64 - cv.meta.n_bits_for_layer_id - bits_per_dim
    y_offset = x_offset - bits_per_dim
    z_offset = y_offset - bits_per_dim

    ids = np.array(ids, dtype=int)
    X = ids >> x_offset & 2**bits_per_dim - 1
    Y = ids >> y_offset & 2**bits_per_dim - 1
    Z = ids >> z_offset & 2**bits_per_dim - 1
    return np.column_stack((X, Y, Z))
