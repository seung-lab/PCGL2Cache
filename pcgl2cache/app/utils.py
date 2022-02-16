import os
from typing import Iterable

import numpy as np
from flask import current_app
from cloudvolume import CloudVolume
from google.auth.credentials import Credentials
from kvdbclient import BigTableClient

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


def get_l2cache_client(graph_id: str) -> BigTableClient:
    l2cache_config = current_app.config["L2CACHE_CONFIG"]
    assert graph_id in l2cache_config, f"Dataset {graph_id} does not have an L2 Cache."

    l2cache_id = l2cache_config[graph_id]["l2cache_id"]
    return BigTableClient(l2cache_id)


def get_l2cache_cv(graph_id: str) -> CloudVolume:
    l2cache_config = current_app.config["L2CACHE_CONFIG"]
    assert (
        graph_id in l2cache_config
    ), f"Dataset {graph_id} does not have CV graphene path."

    cv_path = l2cache_config[graph_id]["cv_path"]
    return CloudVolume(cv_path)


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


def get_registered_attributes() -> dict:
    from ..core import attributes

    attrs = {
        attr.key.decode(): attr for attr in attributes.Attribute._attributes.values()
    }
    attrs.pop("meta")
    return attrs


def get_chunk_coordinates(cv: CloudVolume, ids: Iterable) -> Iterable:
    if not len(ids):
        return np.array([])
    ids = np.array(ids)
    layer = cv.get_chunk_layer(ids[0])
    bits_per_dim = cv.meta.spatial_bit_count[layer]

    x_offset = 64 - cv.meta.n_bits_for_layer_id - bits_per_dim
    y_offset = x_offset - bits_per_dim
    z_offset = y_offset - bits_per_dim

    ids = np.array(ids, dtype=int)
    X = ids >> x_offset & 2**bits_per_dim - 1
    Y = ids >> y_offset & 2**bits_per_dim - 1
    Z = ids >> z_offset & 2**bits_per_dim - 1
    return np.column_stack((X, Y, Z))


def add_coordinates_offset():
    pass
