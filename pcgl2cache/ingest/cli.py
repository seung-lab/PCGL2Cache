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


@ingest_cli.command("atomic")
@click.argument("graph_id", type=str)
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--raw", is_flag=True)
@click.option("--overwrite", is_flag=True, help="Overwrite existing graph")
@click.option("--test", is_flag=True)
def ingest_graph(
    graph_id: str, dataset: click.Path, overwrite: bool, raw: bool, test: bool
):
    """
    Main ingest command
    Takes ingest config from a yaml file and queues atomic tasks
    """
    from . import IngestConfig
    from . import ClusterIngestConfig
    from .cluster import enqueue_atomic_tasks
    from ..backend import BigTableConfig
    from ..backend import DataSource
    from ..backend import GraphConfig
    from ..backend import ChunkedGraphMeta

    with open(dataset, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ingest_config = IngestConfig(
        **config["ingest_config"],
        CLUSTER=ClusterIngestConfig(FLUSH_REDIS=True),
        USE_RAW_EDGES=raw,
        USE_RAW_COMPONENTS=raw,
        TEST_RUN=test,
    )

    graph_config = GraphConfig(
        graph_id=graph_id,
        chunk_size=np.array([256, 256, 512], dtype=int),
        overwrite=True,
    )

    data_source = DataSource(
        agglomeration=config["ingest_config"]["AGGLOMERATION"],
        watershed=config["data_source"]["WATERSHED"],
        edges=config["data_source"]["EDGES"],
        components=config["data_source"]["COMPONENTS"],
        data_version=config["data_source"]["DATA_VERSION"],
        use_raw_edges=raw,
        use_raw_components=raw,
    )

    meta = ChunkedGraphMeta(data_source, graph_config, BigTableConfig())
    enqueue_atomic_tasks(IngestionManager(ingest_config, meta))


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
