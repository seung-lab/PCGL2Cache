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
@click.argument("graph_id", type=str)
@click.argument("cv_path", type=str)
@click.option("--test", is_flag=True)
def ingest_graph(graph_id: str, cv_path: str, test: bool):
    """
    Main ingest command
    Takes ingest config from a yaml file and queues atomic tasks
    """
    from . import IngestConfig
    from .v1.jobs import enqueue_atomic_tasks

    enqueue_atomic_tasks(
        IngestionManager(IngestConfig(TEST_RUN=test), graph_id), cv_path
    )


@ingest_cli.command("status")
def ingest_status():
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    for layer in range(2, imanager.cg_meta.layer_count + 1):
        layer_count = redis.hlen(f"{layer}c")
        print(f"{layer}\t: {layer_count}")
    print(imanager.cg_meta.layer_chunk_counts)


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
