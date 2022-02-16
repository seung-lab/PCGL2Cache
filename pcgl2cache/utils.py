from os import environ


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

    yml_path = environ["GRAPH_L2CACHE_CONFIG_PATH"]
    with open(yml_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError("Unable to read l2cache config.")
    return config
