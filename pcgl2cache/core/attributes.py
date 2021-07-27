import numpy as np
from kvdbclient.serializers import NumPyArray
from kvdbclient.serializers import NumPyValue
from kvdbclient.bigtable.attributes import Attribute

UINT64 = np.dtype("uint64").newbyteorder("L")
UINT32 = np.dtype("uint32").newbyteorder("L")
UINT16 = np.dtype("uint16").newbyteorder("L")
FLOAT16 = np.dtype("float16").newbyteorder("L")
FLOAT32 = np.dtype("float32").newbyteorder("L")

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

# components_: ndarray of shape (n_components, n_features)
# Principal axes in feature space, representing the directions of maximum variance in the data.
# The components are sorted by explained_variance_.
PCA = Attribute(
    key=b"pca",
    family_id="0",
    serializer=NumPyArray(dtype=FLOAT16, shape=(-1, 3)),
)

# singular_values_: ndarray of shape (n_components,)
# The singular values corresponding to each of the selected components.
# The singular values are equal to the 2-norms of the n_components variables in the lower-dimensional space.
PCA_VAL = Attribute(
    key=b"pca_val",
    family_id="0",
    serializer=NumPyArray(dtype=FLOAT32, shape=(-1,)),
)
