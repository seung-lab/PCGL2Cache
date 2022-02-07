import logging
import os
import json


class BaseConfig(object):
    DEBUG = False
    TESTING = False
    HOME = os.path.expanduser("~")
    SECRET_KEY = ""

    LOGGING_FORMAT = '{"source":"%(name)s","time":"%(asctime)s","severity":"%(levelname)s","message":"%(message)s"}'
    LOGGING_DATEFORMAT = "%Y-%m-%dT%H:%M:%S.0Z"
    LOGGING_LEVEL = logging.DEBUG

    CHUNKGRAPH_INSTANCE_ID = "pychunkedgraph"
    PROJECT_ID = os.environ.get("PROJECT_ID", None)
    USE_REDIS_JOBS = False
    CHUNKGRAPH_TABLE_ID = ""

    DATASET_CACHE_ID_MAP = {}
    _l2cache_map = os.environ.get("DATASET_CACHE_ID_MAP", "fly_v31=l2cache_fly_v31_v1")
    _l2cache_map = _l2cache_map.split(",")
    for mapping in _l2cache_map:
        dataset_id, l2cache_id = [x.strip() for x in mapping.split("=")]
        DATASET_CACHE_ID_MAP[dataset_id] = l2cache_id


class DevelopmentConfig(BaseConfig):
    """Development configuration."""

    USE_REDIS_JOBS = False
    DEBUG = True
    LOGGING_LEVEL = logging.ERROR


class DockerDevelopmentConfig(DevelopmentConfig):
    """Development configuration."""

    USE_REDIS_JOBS = True
    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"


class DeploymentWithRedisConfig(BaseConfig):
    """Deployment configuration with Redis."""

    USE_REDIS_JOBS = True
    REDIS_HOST = os.environ.get("REDIS_HOST")
    REDIS_PORT = os.environ.get("REDIS_PORT")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"


class TestingConfig(BaseConfig):
    """Testing configuration."""

    TESTING = True
    USE_REDIS_JOBS = False
    PRESERVE_CONTEXT_ON_EXCEPTION = False
