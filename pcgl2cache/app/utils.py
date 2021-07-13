import os

import numpy as np
from flask import current_app, json
from google.auth.credentials import Credentials

from pychunkedgraph.backend import chunkedgraph

CACHE = {}


class DoNothingCreds(Credentials):
    def refresh(self, request):
        pass


def get_app_base_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_instance_folder_path():
    return os.path.join(get_app_base_path(), "instance")


def jsonify_with_kwargs(data, as_response=True, **kwargs):
    from flask import json

    kwargs.setdefault("separators", (",", ":"))
    if current_app.config["JSONIFY_PRETTYPRINT_REGULAR"] or current_app.debug:
        kwargs["indent"] = 2
        kwargs["separators"] = (", ", ": ")

    resp = json.dumps(data, **kwargs)
    if as_response:
        return current_app.response_class(
            resp + "\n", mimetype=current_app.config["JSONIFY_MIMETYPE"]
        )
    else:
        return resp


def get_bigtable_client(config):
    from google.cloud.bigtable import Client
    from google.auth import default as default_creds

    project_id = config.get("PROJECT_ID", None)
    if config.get("emulate", False):
        credentials = DoNothingCreds()
    elif project_id is not None:
        credentials, _ = default_creds()
    else:
        credentials, project_id = default_creds()

    client = Client(admin=True, project=project_id, credentials=credentials)
    return client


def get_cg(table_id):
    assert table_id in current_app.config["PCG_GRAPH_IDS"]

    if table_id not in CACHE:
        import logging
        from sys import stdout
        from time import gmtime
        from pychunkedgraph.logging import jsonformatter

        instance_id = current_app.config["CHUNKGRAPH_INSTANCE_ID"]
        client = get_bigtable_client(current_app.config)

        logger = logging.getLogger(f"{instance_id}/{table_id}")
        logger.setLevel(current_app.config["LOGGING_LEVEL"])

        # prevent duplicate logs from Flasks(?) parent logger
        logger.propagate = False
        handler = logging.StreamHandler(stdout)
        handler.setLevel(current_app.config["LOGGING_LEVEL"])
        formatter = jsonformatter.JsonFormatter(
            fmt=current_app.config["LOGGING_FORMAT"],
            datefmt=current_app.config["LOGGING_DATEFORMAT"],
        )
        formatter.converter = gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Create ChunkedGraph
        CACHE[table_id] = chunkedgraph.ChunkedGraph(
            table_id=table_id, instance_id=instance_id, client=client, logger=logger
        )
    current_app.table_id = table_id
    return CACHE[table_id]


def toboolean(value):
    """Transform value to boolean type.
    :param value: bool/int/str
    :return: bool
    :raises: ValueError, if value is not boolean.
    """
    if not value:
        raise ValueError("Can't convert null to boolean")

    if isinstance(value, bool):
        return value
    try:
        value = value.lower()
    except:
        raise ValueError(f"Can't convert {value} to boolean")

    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False

    raise ValueError(f"Can't convert {value} to boolean")


def tobinary(ids):
    """Transform id(s) to binary format

    :param ids: uint64 or list of uint64s
    :return: binary
    """
    return np.array(ids).tobytes()


def tobinary_multiples(arr):
    """Transform id(s) to binary format

    :param arr: list of uint64 or list of uint64s
    :return: binary
    """
    return [np.array(arr_i).tobytes() for arr_i in arr]


def get_username_dict(user_ids, auth_token):
    import requests

    AUTH_URL = os.environ.get("AUTH_URL", None)
    if AUTH_URL is None:
        from pychunkedgraph.backend.chunkedgraph_exceptions import ChunkedGraphError

        raise ChunkedGraphError("No AUTH_URL defined")

    users_request = requests.get(
        f"https://{AUTH_URL}/api/v1/username?id={','.join(map(str, np.unique(user_ids)))}",
        headers={"authorization": "Bearer " + auth_token},
        timeout=5,
    )
    return {x["id"]: x["name"] for x in users_request.json()}