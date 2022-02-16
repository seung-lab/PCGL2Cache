def get_chunkedgraph(graph_id):
    from os import environ

    if environ.get("CHUNKEDGRAPH_VERSION", "1") == "1":
        from pychunkedgraph.backend.chunkedgraph import ChunkedGraph

        return ChunkedGraph(graph_id)
    from pychunkedgraph.graph import ChunkedGraph

    return ChunkedGraph(graph_id=graph_id)
