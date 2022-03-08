import logging
from collections import Counter

import numpy as np
import fastremap
from edt import edt
from sklearn import decomposition
from kvdbclient import BigTableClient
from cloudvolume import CloudVolume

from . import attributes


class L2ChunkVolume:
    def __init__(self, cv, cg, coordinates, timestamp):
        self._cv = cv
        self._cg = cg
        self._coordinates = coordinates
        self._chunk_size = self.cv.graph_chunk_size
        self._coordinates_sv = coordinates * self.chunk_size
        self._timestamp = timestamp
        self._vol_bounds = self._cv.bounds

    @property
    def cg(self):
        return self._cg

    @property
    def cv(self) -> CloudVolume:
        return self._cv

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def coordinates_sv(self):
        # coordinates in supervoxel space
        return self._coordinates_sv

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def bbox(self):
        return np.array(self._vol_bounds.to_list())

    def get_volume(self):
        vol_start = self.bbox[:3] + self.coordinates_sv
        vol_end = vol_start + self.chunk_size

        return self.cv[
            vol_start[0] : vol_end[0],
            vol_start[1] : vol_end[1],
            vol_start[2] : vol_end[2],
        ][..., 0]

    def get_remapped_segmentation(self, l2id=None):
        """
        Remaps suoervoxel IDs in a chunk volume with L2 parent IDs represented by contiguous IDs.
        """
        vol = self.get_volume()
        svids = fastremap.unique(vol)
        svids = svids[svids != 0]
        if len(svids) == 0:
            return vol.astype(np.uint32), {}

        _l2ids = _get_l2_ids(self, svids)
        if l2id is not None:
            # remap given l2id from get_roots to given l2id
            remapping = {}
            if self.cg is not None:
                children = self.cg.get_children(l2id)
            else:
                children = self.cv.get_leaves(l2id, self._vol_bounds, 0)
            try:
                idx = np.where(svids == children[0])[0][0]
                parent = _l2ids[idx]
                remapping[parent] = l2id
            except IndexError:
                pass
            fastremap.remap(
                _l2ids, remapping, in_place=True, preserve_missing_labels=True
            )
            fastremap.mask_except(vol, children.tolist(), in_place=True)

        u_l2ids = fastremap.unique(_l2ids)
        u_cont_ids = np.arange(1, 1 + len(u_l2ids))
        cont_ids = fastremap.remap(_l2ids, dict(zip(u_l2ids, u_cont_ids)))
        fastremap.remap(
            vol,
            dict(zip(svids, cont_ids)),
            preserve_missing_labels=True,
            in_place=True,
        )
        return vol.astype(np.uint32), dict(zip(u_cont_ids, u_l2ids))


def _get_l2_ids(l2vol: L2ChunkVolume, svids: np.array) -> np.array:
    if l2vol.cg:
        l2ids = l2vol.cg.get_roots(svids, stop_layer=2, time_stamp=l2vol.timestamp)
        layers = l2vol.cg.get_chunk_layers(l2ids)
        sv_mask = layers == 1
        l2ids[sv_mask] = 0
    else:
        l2ids = l2vol.cv.get_roots(svids, timestamp=l2vol.timestamp, stop_layer=2)
    return l2ids


def get_edt_stack(vol_l2, resolution):
    # First calculate eucledian distance transform for all segments
    # Every entrie in vol_dt is the distance in nm from the closest
    # boundary
    vol_dt = edt(
        vol_l2,
        anisotropy=resolution,
        black_border=False,
        parallel=1,  # number of threads, <= 0 sets to num cpu
    )

    # To efficiently map measured distances from the EDT to all IDs
    # we use `fastremap.inverse_component_map`. This function takes
    # two equally sized volumes - the first has the IDs, the second
    # the data we want to map. However, this function uniquenifies
    # the data entries per ID such that we loose the size information.
    # Additionally, we want to retain information about the locations.
    # To enable this with one iteration of the block, we build a
    # compound data block. Each value has 64 bits, the first 32 bits
    # encode the EDT, the second the location as flat index. Using,
    # float data for the edt would lead to overflows, so we first
    # convert to uints.
    shape = np.array(vol_l2.shape)
    size = np.product(shape)
    stack = ((vol_dt.astype(np.uint64).flatten()) << 32) + np.arange(
        size, dtype=np.uint64
    )
    return stack


def process_edt_stack(vol_l2, l2_contiguous_d, edt_stack, l2id=None):
    # cmap_stack is a dictionary of (L2) IDs -> list of 64 bit values
    # encoded as described in `get_edt_stack`.
    if l2id is not None:
        l2_dict_reverse = {v: k for k, v in l2_contiguous_d.items()}
        try:
            l2_cont_id = l2_dict_reverse[l2id]
            l2ids = np.array([l2_cont_id])
        except KeyError:
            logging.warning(f"Unable to process L2 ID {l2id}")
            l2ids = np.array([])
        if l2ids.size == 0:
            return {}
        nonzero_mask = vol_l2.flatten() != 0
        cmap_stack = {l2_cont_id: edt_stack[nonzero_mask]}
    else:
        cmap_stack = fastremap.inverse_component_map(vol_l2.flatten(), edt_stack)
        l2ids = np.array(list(cmap_stack.keys()))
        l2ids = l2ids[l2ids != 0]
    return cmap_stack, l2ids


def dist_weight(resolution, coords):
    import warnings

    mean_coord = np.mean(coords, axis=0)
    dists = np.linalg.norm((coords - mean_coord) * resolution, axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return 1 - dists / dists.max()


def calculate_features(vol_l2, l2_cont_d, resolution, l2id=None):
    edt_stack = get_edt_stack(vol_l2, resolution)
    cmap_stack, l2ids = process_edt_stack(vol_l2, l2_cont_d, edt_stack, l2id=l2id)

    pca = decomposition.PCA(3)
    l2_max_coords = []
    l2_max_scaled_coords = []
    l2_bboxs = []
    l2_chunk_intersects = []
    l2_max_dts = []
    l2_mean_dts = []
    l2_sizes = []
    l2_pca_comps = []
    l2_pca_vals = []
    for l2id in l2ids:
        # We first disentangle the compound data for the specific L2 ID
        # and transform the flat indices to 3d indices.
        l2_stack = np.array(cmap_stack[l2id], dtype=np.uint64)
        dts = l2_stack >> 32
        idxs = l2_stack.astype(np.uint32)
        coords = np.array(np.unravel_index(np.array(idxs), vol_l2.shape)).T

        # Finally, we compute statistics from the decoded data.
        max_idx = np.argmax(dts)
        l2_max_coords.append(coords[max_idx])
        l2_max_scaled_coords.append(
            coords[np.argmax(dts * dist_weight(resolution, coords))]
        )
        l2_bboxs.append([np.min(coords, axis=0), np.max(coords, axis=0)])
        l2_sizes.append(len(idxs))
        l2_max_dts.append(dts[max_idx])
        l2_mean_dts.append(np.mean(dts))
        l2_chunk_intersects.append(
            [
                np.sum(coords == 0, axis=0),
                np.sum((coords + 1 - vol_l2.shape) == 0, axis=0),
            ]
        )

        # for consistency use biological size 0.01 um^3 for filtering small objects
        if len(idxs) * np.product(resolution) / 1e9 < 0.01:
            l2_pca_comps.append(np.zeros(shape=(0, 3), dtype=attributes.PCA.basetype))
            l2_pca_vals.append(np.zeros(shape=(0,), dtype=attributes.PCA_VAL.basetype))
            continue

        # The PCA calculation is straight-forward as long as the are sufficiently
        # many coordinates. We observed long runtimes for very large components.
        # Using a subset of the points in such cases proved to produce almost
        # identical results.
        if len(coords) < 3:
            coords_p = np.concatenate([coords, coords, coords])
        elif len(coords) > 10000:
            coords_p = np.array(
                np.unravel_index(
                    np.random.choice(idxs, 10000, replace=False), vol_l2.shape
                )
            ).T
        else:
            coords_p = coords

        pca.fit(coords_p * resolution)
        comps = np.array(pca.components_, dtype=attributes.PCA.basetype)
        vals = np.array(pca.singular_values_, dtype=attributes.PCA_VAL.basetype)
        l2_pca_comps.append(comps)
        l2_pca_vals.append(vals)

    l2_sizes = np.array(l2_sizes)
    l2_max_dts = np.array(l2_max_dts)
    l2_mean_dts = np.array(l2_mean_dts)

    l2_max_scaled_coords = np.array(l2_max_scaled_coords)
    l2_chunk_intersects = np.array(l2_chunk_intersects)

    # Area calculations are handled seaprately and are performed by overlap through
    # shifts. We shift in each dimension and calculate the overlapping segment ids.
    # The overlapping IDs are then counted per dimension and added up after
    # adjusting for resolution. This measurement will overestimate area slightly
    # but smoothed measurements are ill-defined and too compute intensive.
    x_m = vol_l2[1:] != vol_l2[:-1]
    y_m = vol_l2[:, 1:] != vol_l2[:, :-1]
    z_m = vol_l2[:, :, 1:] != vol_l2[:, :, :-1]

    u_x, c_x = fastremap.unique(
        np.concatenate([vol_l2[1:][x_m], vol_l2[:-1][x_m]]), return_counts=True
    )
    u_y, c_y = fastremap.unique(
        np.concatenate([vol_l2[:, 1:][y_m], vol_l2[:, :-1][y_m]]), return_counts=True
    )
    u_z, c_z = fastremap.unique(
        np.concatenate([vol_l2[:, :, 1:][z_m], vol_l2[:, :, :-1][z_m]]),
        return_counts=True,
    )

    x_area = np.product(resolution[[1, 2]])
    y_area = np.product(resolution[[0, 2]])
    z_area = np.product(resolution[[0, 1]])

    x_dict = Counter(dict(zip(u_x, c_x * x_area)))
    y_dict = Counter(dict(zip(u_y, c_y * y_area)))
    z_dict = Counter(dict(zip(u_z, c_z * z_area)))

    area_dict = x_dict + y_dict + z_dict
    areas = np.array([area_dict[l2id] for l2id in l2ids])

    return {
        "l2id": fastremap.remap(l2ids, l2_cont_d).astype(attributes.UINT64.type),
        "size_nm3": l2_sizes.astype(attributes.SIZE_NM3.basetype),
        "area_nm2": areas.astype(attributes.AREA_NM2.basetype),
        "max_dt_nm": l2_max_dts.astype(attributes.MAX_DT_NM.basetype),
        "mean_dt_nm": l2_mean_dts.astype(attributes.MEAN_DT_NM.basetype),
        "rep_coord_nm": l2_max_scaled_coords.astype(attributes.REP_COORD_NM.basetype),
        "chunk_intersect_count": l2_chunk_intersects.astype(
            attributes.CHUNK_INTERSECT_COUNT.basetype
        ),
        "pca_comp": l2_pca_comps,
        "pca_vals": l2_pca_vals,
    }


def run_l2cache(
    cv: CloudVolume, cg=None, chunk_coord=None, timestamp=None, l2id=None
) -> dict:
    if chunk_coord is None:
        assert l2id is not None
        from ..utils import get_chunk_coordinates

        _coords = get_chunk_coordinates(cv, [l2id])
        chunk_coord = _coords[0]

    l2chunk = L2ChunkVolume(cv, cg, np.array(list(chunk_coord), dtype=int), timestamp)
    vol_l2, l2_contiguous_d = l2chunk.get_remapped_segmentation(l2id)
    if np.sum(np.array(list(l2_contiguous_d.values())) != 0) == 0:
        return {}
    return calculate_features(vol_l2, l2_contiguous_d, cv.resolution, l2id)


def write_to_db(client: BigTableClient, result_d: dict) -> None:
    from kvdbclient.base import Entry
    from kvdbclient.base import EntryKey

    entries = []
    for tup in zip(*result_d.values()):
        (
            l2id,
            size_nm3,
            area_nm2,
            max_dt_nm,
            mean_dt_nm,
            rep_coord_nm,
            chunk_intersect_count,
            pca_comp,
            pca_vals,
        ) = tup
        val_d = {
            attributes.SIZE_NM3: size_nm3,
            attributes.AREA_NM2: area_nm2,
            attributes.MAX_DT_NM: max_dt_nm,
            attributes.MEAN_DT_NM: mean_dt_nm,
            attributes.REP_COORD_NM: rep_coord_nm,
            attributes.CHUNK_INTERSECT_COUNT: chunk_intersect_count,
            attributes.PCA: pca_comp,
            attributes.PCA_VAL: pca_vals,
        }
        entries.append(Entry(EntryKey(l2id), val_d))
    client.write_entries(entries)
