"""
cli for running ingest
"""

import yaml
import numpy as np
import click
from flask.cli import AppGroup

from .manager import IngestionManager
from .redis import keys as r_keys
from .redis import get_redis_connection

ingest_cli = AppGroup("ingest")


@ingest_cli.command("v1")
@click.argument("cache_id", type=str)
@click.argument("graph_id", type=str)
@click.argument("cv_path", type=str)
@click.option("--test", is_flag=True)
def ingest_graph(cache_id: str, graph_id: str, cv_path: str, test: bool):
    """
    Main ingest command
    Takes ingest config from a yaml file and queues atomic tasks
    """
    from . import IngestConfig
    from . import ClusterIngestConfig
    from .v1.jobs import enqueue_atomic_tasks

    enqueue_atomic_tasks(
        IngestionManager(
            IngestConfig(CLUSTER=ClusterIngestConfig(), TEST_RUN=test),
            cache_id,
            graph_id,
        ),
        cv_path,
    )


@ingest_cli.command("status")
def ingest_status():
    redis = get_redis_connection()
    print(redis.hlen("2c"))


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
