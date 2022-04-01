import json
import time
import traceback
import os
from datetime import datetime
from typing import Iterable

import numpy as np
from pytz import UTC
from cloudvolume import compression

from flask import g
from flask import request
from flask import jsonify
from flask import Blueprint
from flask import current_app
from flask import make_response

from .utils import get_registered_attributes

__api_versions__ = [1]
__pcgl2cache_url_prefix__ = os.environ.get("PCGL2CACHE_URL_PREFIX", "l2cache")


bp = Blueprint(
    "pcgl2cache",
    __name__,
    url_prefix=f"/{__pcgl2cache_url_prefix__}",
)


@bp.route("/")
@bp.route("/index")
def index():
    from .. import __version__

    return f"PCGL2Cache v{__version__}"


def home():
    resp = make_response()
    resp.headers["Access-Control-Allow-Origin"] = "*"
    acah = "Origin, X-Requested-With, Content-Type, Accept"
    resp.headers["Access-Control-Allow-Headers"] = acah
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Connection"] = "keep-alive"
    return resp


def before_request():
    current_app.request_start_time = time.time()
    current_app.request_start_date = datetime.utcnow()
    current_app.user_id = None
    current_app.table_id = None
    current_app.request_type = None
    content_encoding = request.headers.get("Content-Encoding", "")
    if "gzip" in content_encoding.lower():
        request.data = compression.decompress(request.data, "gzip")


def after_request(response):
    dt = (time.time() - current_app.request_start_time) * 1000
    current_app.logger.debug("Response time: %.3fms" % dt)
    accept_encoding = request.headers.get("Accept-Encoding", "")
    if "gzip" not in accept_encoding.lower():
        return response

    response.direct_passthrough = False
    if (
        response.status_code < 200
        or response.status_code >= 300
        or "Content-Encoding" in response.headers
    ):
        return response

    response.data = compression.gzip_compress(response.data)
    response.headers["Content-Encoding"] = "gzip"
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["Content-Length"] = len(response.data)
    return response


def unhandled_exception(e):
    status_code = 500
    response_time = (time.time() - current_app.request_start_time) * 1000
    user_ip = str(request.remote_addr)
    tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
    current_app.logger.error(
        {
            "message": str(e),
            "user_id": user_ip,
            "user_ip": user_ip,
            "request_time": current_app.request_start_date,
            "request_url": request.url,
            "request_data": request.data,
            "response_time": response_time,
            "response_code": status_code,
            "traceback": tb,
        }
    )
    resp = {
        "timestamp": current_app.request_start_date,
        "duration": response_time,
        "code": status_code,
        "message": str(e),
        "traceback": tb,
    }
    return jsonify(resp), status_code


def api_exception(e):
    response_time = (time.time() - current_app.request_start_time) * 1000
    user_ip = str(request.remote_addr)
    tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
    current_app.logger.error(
        {
            "message": str(e),
            "user_id": user_ip,
            "user_ip": user_ip,
            "request_time": current_app.request_start_date,
            "request_url": request.url,
            "request_data": request.data,
            "response_time": response_time,
            "response_code": e.status_code.value,
            "traceback": tb,
        }
    )
    resp = {
        "timestamp": current_app.request_start_date,
        "duration": response_time,
        "code": e.status_code.value,
        "message": str(e),
    }
    return jsonify(resp), e.status_code.value


def handle_attr_metadata():
    return {
        name: str(attr.serializer.basetype)
        for name, attr in get_registered_attributes().items()
    }


def handle_attributes(graph_id: str, is_binary=False):
    from ..app.utils import get_l2cache_client

    if is_binary:
        l2ids = np.frombuffer(request.data, np.uint64)
    else:
        l2ids = np.array(json.loads(request.data)["l2_ids"], dtype=np.uint64)

    attributes = None
    attribute_names = request.args.get("attribute_names")
    if attribute_names is not None:
        attribute_names = [x.strip() for x in attribute_names.split(",")]
        _attributes = get_registered_attributes()
        attributes = []
        for name in attribute_names:
            # assert name in _attributes
            attributes.append(_attributes[name])

    cache_client = get_l2cache_client(graph_id)
    entries = cache_client.read_entries(keys=l2ids, attributes=attributes)
    result = {}
    missing_l2ids = []
    for l2id in l2ids:
        try:
            attrs = entries[l2id]
            result[int(l2id)] = {}
            for k, v in attrs.items():
                val = v[0].value
                try:
                    # if empty list skip from response
                    if len(val) > 0:
                        result[int(l2id)][k.decode()] = val
                except TypeError:
                    # add all scalar values to response
                    result[int(l2id)][k.decode()] = val
        except KeyError:
            result[int(l2id)] = {}
            missing_l2ids.append(l2id)
    _add_offset_to_coords(graph_id, l2ids, result)

    if "size_nm3" in attributes:
        _rescale_volume(graph_id, l2ids, result)

    try:
        _trigger_cache_update(missing_l2ids, graph_id, cache_client.table_id)
    except Exception as e:
        current_app.logger.error(str(e))
    return result


def _rescale_volume(graph_id: str, l2ids: Iterable, result: dict):
    from .utils import get_l2cache_cv

    # Get volume of a supervoxel in nm3
    cv = get_l2cache_cv(graph_id)
    sv_vol = np.array(cv.mip_resolution(0)).prod()

    for l2id in l2ids:
        key = int(l2id)
        try:
            features = result[key]
            vol = features["size_nm3"]
            result[key]["size_nm3"] = vol * sv_vol
        except KeyError:
            continue


def _add_offset_to_coords(graph_id: str, l2ids: Iterable, result: dict):
    from .utils import get_l2cache_cv
    from ..utils import get_chunk_coordinates

    cv = get_l2cache_cv(graph_id)
    start_offset = np.array(cv.bounds.to_list()[:3])
    coords = get_chunk_coordinates(cv, l2ids)

    for l2id, coord in zip(l2ids, coords):
        key = int(l2id)
        try:
            features = result[key]
        except KeyError:
            continue

        try:
            rep_coord = features["rep_coord_nm"]
            rep_coord = np.array(rep_coord, dtype=np.uint64)
            offset = (coord * cv.graph_chunk_size) + start_offset
            rep_coord = (rep_coord + offset) * cv.resolution
            result[key]["rep_coord_nm"] = rep_coord
        except KeyError:
            continue


def _trigger_cache_update(l2ids, graph_id: str, l2_cache_id: str) -> None:
    import numpy as np
    from messagingclient import MessagingClient

    payload = np.array(l2ids, dtype=np.uint64).tobytes()
    attributes = {
        "table_id": graph_id,
        "l2_cache_id": l2_cache_id,
    }

    c = MessagingClient()
    exchange = os.getenv("L2CACHE_EXCHANGE", "pychunkedgraph")
    c.publish(exchange, payload, attributes)
