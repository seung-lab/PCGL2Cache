from collections import namedtuple

from .bigtable.client import Client as BigTableClient


_backend_clientinfo_fields = ("TYPE", "CONFIG")
_backend_clientinfo_defaults = (None, None)
BackendClientInfo = namedtuple(
    "BackendClientInfo",
    _backend_clientinfo_fields,
    defaults=_backend_clientinfo_defaults,
)


def get_default_client_info():
    """
    Load client from env variables.
    """

    # TODO make dynamic after multiple platform support is added
    from .bigtable import get_client_info as get_bigtable_client_info

    return BackendClientInfo(
        CONFIG=get_bigtable_client_info(admin=True, read_only=False)
    )
