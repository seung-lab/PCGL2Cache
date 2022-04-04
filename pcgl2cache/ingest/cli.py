"""
cli for running ingest
"""

import click
from flask.cli import AppGroup

from .manager import IngestionManager
from .redis import get_redis_connection

ingest_cli = AppGroup("ingest")


@ingest_cli.command("flush_redis")
def flush_redis():
    """FLush redis db."""
    redis = get_redis_connection()
    redis.flushdb()


@ingest_cli.command("v1")
@click.argument("cache_id", type=str)
@click.argument("graph_id", type=str)
@click.argument("cv_path", type=str)
@click.argument("timestamp", type=str)
@click.option("--create", is_flag=True)
@click.option("--test", is_flag=True)
def ingest_cache(
    cache_id: str, graph_id: str, cv_path: str, timestamp: str, create: bool, test: bool
):
    """
    Main ingest command
    Takes ingest config from a yaml file and queues atomic tasks
    """
    from datetime import datetime
    from . import IngestConfig
    from . import ClusterIngestConfig
    from .v1.jobs import enqueue_atomic_tasks

    if create:
        from kvdbclient import BigTableClient

        client = BigTableClient()
        client.create_table(cache_id)

    # example format Jun 1 2005 1:33PM
    timestamp = datetime.strptime(timestamp, "%b %d %Y %I:%M%p")
    enqueue_atomic_tasks(
        IngestionManager(
            IngestConfig(CLUSTER=ClusterIngestConfig(), TEST_RUN=test),
            cache_id,
            graph_id,
        ),
        cv_path,
        timestamp,
    )


@ingest_cli.command("v2")
@click.argument("cache_id", type=str)
@click.argument("graph_id", type=str)
@click.argument("cv_path", type=str)
@click.argument("timestamp", type=str)
@click.option("--create", is_flag=True)
@click.option("--test", is_flag=True)
def ingest_cache_v2(
    cache_id: str, graph_id: str, cv_path: str, timestamp: str, create: bool, test: bool
):
    """
    Main ingest command
    Takes ingest config from a yaml file and queues atomic tasks
    """
    from datetime import datetime
    from . import IngestConfig
    from . import ClusterIngestConfig
    from .v2.jobs import enqueue_atomic_tasks

    if create:
        from kvdbclient import get_default_client_info
        from kvdbclient import BigTableClient

        client = BigTableClient(config=get_default_client_info())
        client.create_table(cache_id)

    # example format Jun 1 2005 1:33PM
    timestamp = datetime.strptime(timestamp, "%b %d %Y %I:%M%p")
    enqueue_atomic_tasks(
        IngestionManager(
            IngestConfig(CLUSTER=ClusterIngestConfig(), TEST_RUN=test),
            cache_id,
            graph_id,
        ),
        cv_path,
        timestamp,
    )


@ingest_cli.command("status")
def ingest_status():
    redis = get_redis_connection()
    print(redis.scard("2c"))


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
