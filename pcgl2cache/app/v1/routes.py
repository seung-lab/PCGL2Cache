from flask import request, current_app
from flask import Blueprint
from middle_auth_client import auth_required
from middle_auth_client import auth_requires_permission
from functools import wraps

from ...app import common
from ..utils import toboolean
from ..utils import jsonify_with_kwargs
from ...utils import read_l2cache_config

bp = Blueprint(
    "pcgl2cache_v1",
    __name__,
    url_prefix=f"/{common.__pcgl2cache_url_prefix__}/api/v1",
)


def remap_public(func=None, *, check_node_ids=False):
    def mydecorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            virtual_tables = current_app.config.get("VIRTUAL_TABLES", None)

            # if not virtual configuration just return
            if virtual_tables is None:
                return f(*args, **kwargs)
            table_id = kwargs.get("table_id", None)

            if table_id is None:
                # then no table remapping necessary
                return f(*args, **kwargs)
            if not table_id in virtual_tables:
                # if table table_id isn't in virtual
                # tables then just return
                return f(*args, **kwargs)

            # then we have a virtual table
            # and we want to remap the table name
            new_table = virtual_tables[table_id]["table_id"]
            kwargs["table_id"] = new_table

            return f(*args, **kwargs)

        return decorated_function

    if func:
        return mydecorator(func)
    else:
        return mydecorator


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


@bp.route("/attribute_metadata", methods=["GET"])
def attr_metadata():
    return jsonify_with_kwargs(common.handle_attr_metadata())


@bp.route("/table/<table_id>/attributes", methods=["POST"])
@auth_requires_permission("view")
@remap_public
def attributes(table_id):
    import gc

    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    result = jsonify_with_kwargs(
        common.handle_attributes(table_id), int64_as_str=int64_as_str
    )
    gc.collect()
    return result


@bp.route("/table_mapping", methods=["GET"])
def attr_metadata():
    return jsonify_with_kwargs(read_l2cache_config())
