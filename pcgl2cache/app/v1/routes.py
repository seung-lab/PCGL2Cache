from flask import request
from flask import Blueprint
from middle_auth_client import auth_required
from middle_auth_client import auth_requires_permission
from pychunkedgraph.backend.chunkedgraph_exceptions import ChunkedGraphAPIError

from ...app import common
from ...app.utils import tobinary
from ...app.utils import toboolean
from ...app.utils import jsonify_with_kwargs

bp = Blueprint(
    "pcgl2cache_v1",
    __name__,
    url_prefix=f"/{common.__pcgl2cache_url_prefix__}/api/v1",
)


@bp.route("/")
@bp.route("/index")
@auth_required
def index():
    return common.index()


@bp.route
@auth_required
def home():
    return common.home()


@bp.before_request
def before_request():
    return common.before_request()


@bp.after_request
def after_request(response):
    return common.after_request(response)


@bp.errorhandler(Exception)
def unhandled_exception(e):
    return common.unhandled_exception(e)


@bp.errorhandler(ChunkedGraphAPIError)
def api_exception(e):
    return common.api_exception(e)


@bp.route("/attribute_metadata", methods=["GET"])
def attr_metadata():
    return jsonify_with_kwargs(common.handle_attr_metadata())


@bp.route("/table/<table_id>/attributes", methods=["POST"])
def attributes(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    return jsonify_with_kwargs(
        common.handle_attributes(table_id), int64_as_str=int64_as_str
    )
