import pickle
from typing import Dict

from rq import Queue as RQueue


from . import IngestConfig


def _get_cg(graph_id):
    from os import environ

    if environ.get("CHUNKEDGRAPH_VERSION", "1") == "1":
        from pychunkedgraph.backend.chunkedgraph import ChunkedGraph

        return ChunkedGraph(graph_id)
    from pychunkedgraph.graph import ChunkedGraph

    return ChunkedGraph(graph_id=graph_id)


class IngestionManager:
    def __init__(self, config: IngestConfig, cache_id: str, graph_id: str):
        self._config = config
        self._cg = None
        self._cache_id = cache_id
        self._graph_id = graph_id
        self._redis = None
        self._task_queues = {}

    @property
    def config(self):
        return self._config

    @property
    def cache_id(self):
        return self._cache_id

    @property
    def graph_id(self):
        return self._graph_id

    @property
    def cg(self):
        if self._cg is None:
            self._cg = _get_cg(self._graph_id)
        return self._cg

    @property
    def redis(self):
        if self._redis is not None:
            return self._redis
        from .redis import get_redis_connection
        from .redis import keys as r_keys

        self._redis = get_redis_connection()
        self._redis.set(r_keys.INGESTION_MANAGER, self.serialize_info(pickled=True))
        return self._redis

    @classmethod
    def from_pickle(cls, serialized_info):
        return cls(**pickle.loads(serialized_info))

    def serialize_info(self, pickled=False):
        info = {
            "config": self._config,
            "cache_id": self._cache_id,
            "graph_id": self._graph_id,
        }
        if pickled:
            return pickle.dumps(info)
        return info

    def get_task_queue(self, q_name) -> RQueue:
        from .redis import get_rq_queue

        if q_name in self._task_queues:
            return self._task_queues[q_name]
        self._task_queues[q_name] = get_rq_queue(q_name)
        return self._task_queues[q_name]
