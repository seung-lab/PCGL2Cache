import io
import pickle
import pandas as pd

from flask import make_response, current_app
from flask import Blueprint, request
from middle_auth_client import auth_requires_permission
from middle_auth_client import auth_requires_admin
from middle_auth_client import auth_required

from pcgl2cache.app.app_utils import jsonify_with_kwargs, toboolean, tobinary
from pcgl2cache.app import common

bp = Blueprint(
    "pcgl2cache_v1",
    __name__,
    url_prefix=f"/{common.__pcgl2cache_url_prefix__}/api/v1",
)

# -------------------------------
# ------ Access control and index
# -------------------------------


@bp.route("/")
@bp.route("/index")
@auth_required
def index():
    return common.index()


@bp.route
@auth_required
def home():
    return common.home()


# -------------------------------
# ------ Measurements and Logging
# -------------------------------


@bp.before_request
# @auth_required
def before_request():
    return common.before_request()


@bp.after_request
# @auth_required
def after_request(response):
    return common.after_request(response)


@bp.errorhandler(Exception)
def unhandled_exception(e):
    return common.unhandled_exception(e)


@bp.errorhandler(cg_exceptions.ChunkedGraphAPIError)
def api_exception(e):
    return common.api_exception(e)


