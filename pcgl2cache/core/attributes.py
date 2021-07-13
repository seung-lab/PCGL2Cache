import numpy as np
from kvdbclient.serializers import NumPyArray
from kvdbclient.serializers import NumPyValue
from kvdbclient.bigtable.attributes import Attribute

UINT64 = np.dtype("uint64").newbyteorder("L")
UINT32 = np.dtype("uint32").newbyteorder("L")
UINT16 = np.dtype("uint16").newbyteorder("L")
FLOAT16 = np.dtype("float16").newbyteorder("L")

SIZE_NM3 = Attribute(
    key=b"size_nm3", family_id="0", serializer=NumPyValue(dtype=UINT32)
)

AREA_NM2 = Attribute(
    key=b"area_nm2", family_id="0", serializer=NumPyValue(dtype=UINT32)
)

MAX_DT_NM = Attribute(
    key=b"max_dt_nm", family_id="0", serializer=NumPyValue(dtype=UINT16)
)

MEAN_DT_NM = Attribute(
    key=b"mean_dt_nm", family_id="0", serializer=NumPyValue(dtype=FLOAT16)
)

REP_COORD_NM = Attribute(
    key=b"rep_coord_nm",
    family_id="0",
    serializer=NumPyArray(dtype=UINT64),
)

CHUNK_INTERSECT_COUNT = Attribute(
    key=b"chunk_intersect_count",
    family_id="0",
    serializer=NumPyArray(dtype=UINT16, shape=(-1, 3)),
)

PCA = Attribute(
    key=b"pca",
    family_id="0",
    serializer=NumPyArray(dtype=FLOAT16, shape=(-1, 3)),
)
