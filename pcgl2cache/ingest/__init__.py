from os import environ
from collections import namedtuple
from os import environ

from .redis import REDIS_URL


_cluster_ingest_config_fields = (
    "REDIS_URL",
    "BATCH_SIZE",
    "L2CACHE_Q_NAME",
    "L2CACHE_Q_LIMIT",  # these limits ensure the queue won't use too much memory
    "L2CACHE_Q_INTERVAL",  # sleep interval before queuing the next job when limit is reached
)
_cluster_ingest_defaults = (
    REDIS_URL,
    10,
    "l2",
    int(environ.get("L2CACHE_Q_LIMIT", 100000)),
    60,
)
ClusterIngestConfig = namedtuple(
    "ClusterIngestConfig",
    _cluster_ingest_config_fields,
    defaults=_cluster_ingest_defaults,
)


_ingestconfig_fields = (
    "CLUSTER",  # run ingest on a single machine (simple) or on a cluster
    "AGGLOMERATION",
    "WATERSHED",
    "USE_RAW_EDGES",
    "USE_RAW_COMPONENTS",
    "TEST_RUN",
)
_ingestconfig_defaults = (None, None, None, False, False, False)
IngestConfig = namedtuple(
    "IngestConfig", _ingestconfig_fields, defaults=_ingestconfig_defaults
)
