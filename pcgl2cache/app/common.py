import collections
import json
import threading
import time
import traceback
import gzip
import os
from io import BytesIO as IO
from datetime import datetime
import requests

import numpy as np
from pytz import UTC
import pandas as pd

from cloudvolume import compression
from middle_auth_client import get_usernames

from flask import current_app, g, jsonify, make_response, request
from pcgl2cache import __version__

from .utils import get_registered_attributes

__api_versions__ = [1]
__pcgl2cache_url_prefix__ = os.environ.get("PCGL2CACHE_URL_PREFIX", "l2cache")


def index():
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


def handle_attributes(table_id: str, is_binary=False):
    # TODO remove test url
    # /l2cache/api/v1/table/fly_v31/attributes?attribute_names=size_nm3,mean_dt_nm
    from ..app.utils import get_l2cache_client

    if is_binary:
        l2_ids = np.frombuffer(request.data, np.uint64)
    else:
        l2_ids = np.array(json.loads(request.data)["l2_ids"], dtype=np.uint64)

    # l2_ids = [
    #     175137943013294113,
    #     175137943013294114,
    #     175137943013294118,
    #     175137943013294119,
    # ]

    attributes = None
    attribute_names = request.args.get("attribute_names")
    if attribute_names is not None:
        attribute_names = [x.strip() for x in attribute_names.split(",")]
        _attributes = get_registered_attributes()
        attributes = []
        for name in attribute_names:
            # assert name in _attributes
            attributes.append(_attributes[name])

    cache_client = get_l2cache_client(table_id)
    entries = cache_client.read_entries(keys=l2_ids, attributes=attributes)
    result = {}
    for l2id in l2_ids:
        try:
            attrs = entries[l2id]
            result[str(l2id)] = {k.decode(): v[0].value for k, v in attrs.items()}
        except KeyError:
            result[str(l2id)] = {}
    return result
