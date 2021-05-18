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
from pcgl2cache.app import app_utils

__api_versions__ = [1]
__segmentation_url_prefix__ = os.environ.get("PCGL2CACHE_URL_PREFIX", "pcgl2cache")


def index():
    return f"PyChunkedGraph Segmentation v{__version__}"


def home():
    resp = make_response()
    resp.headers["Access-Control-Allow-Origin"] = "*"
    acah = "Origin, X-Requested-With, Content-Type, Accept"
    resp.headers["Access-Control-Allow-Headers"] = acah
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Connection"] = "keep-alive"
    return resp


# -------------------------------
# ------ Measurements and Logging
# -------------------------------


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

    try:
        if current_app.user_id is None:
            user_id = ""
        else:
            user_id = current_app.user_id

        if current_app.table_id is not None:
            log_db = app_utils.get_log_db(current_app.table_id)
            log_db.add_success_log(
                user_id=user_id,
                user_ip="",
                request_time=current_app.request_start_date,
                response_time=dt,
                url=request.url,
                request_data=request.data,
                request_type=current_app.request_type,
            )
    except Exception as e:
        current_app.logger.debug(
            f"{current_app.user_id}: LogDB entry not" f" successful: {e}"
        )

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

