import logging
from collections import Counter
from collections import defaultdict

import numpy as np
import fastremap
from edt import edt
from sklearn import decomposition
from kvdbclient import BigTableClient

from memory_profiler import profile


class L2ChunkVolume:
    def __init__(self, cg, cv, coordinates, timestamp):
        self._cg = cg
        self._cv = cv
        self._coordinates = coordinates * self.chunk_size
        self._timestamp = timestamp

    @property
    def cg(self):
        return self._cg

    @property
    def cv(self):
        return self._cv

    @property
    def chunk_size(self):
        return self._cg.chunk_size.astype(np.int)

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def bbox(self):
        return np.array(self.cv.bounds.to_list())

    def get_volume(self):
        vol_start = self.bbox[:3] + self.coordinates
        vol_end = vol_start + self.chunk_size

        return self.cv[
            vol_start[0] : vol_end[0],
            vol_start[1] : vol_end[1],
            vol_start[2] : vol_end[2],
        ][..., 0]

    @profile
    def get_remapped_segmentation(self, l2id=None):
        """
        Remaps suoervoxel IDs in a chunk volume with L2 parent IDs represented by contiguous IDs.
        """
        vol = self.get_volume()
        sv_ids = fastremap.unique(vol)
        sv_ids = sv_ids[sv_ids != 0]
        if len(sv_ids) == 0:
            return vol.astype(np.uint32), {}

        _l2ids = self.cg.get_roots(sv_ids, stop_layer=2, time_stamp=self.timestamp)
        if l2id is not None:
            # remap given l2id from get_roots to given l2id
            remapping = {}
            children = self.cg.get_children(l2id)
            try:
                idx = np.where(sv_ids == children[0])[0][0]
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
            dict(zip(sv_ids, cont_ids)),
            preserve_missing_labels=True,
            in_place=True,
        )
        return vol.astype(np.uint32), dict(zip(u_cont_ids, u_l2ids))


def dist_weight(cv, coords):
    mean_coord = np.mean(coords, axis=0)
    dists = np.linalg.norm((coords - mean_coord) * cv.resolution, axis=1)
    return 1 - dists / dists.max()


@profile
def calculate_features(cv, chunk_coord, vol_l2, l2_contiguous_d, l2id=None):
    from . import attributes

    # First calculate eucledian distance transform for all segments
    # Every entrie in vol_dt is the distance in nm from the closest
    # boundary
    vol_dt = edt(
        vol_l2,
        anisotropy=cv.resolution,
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

    # cmap_stack is a dictionary of (L2) IDs -> list of 64 bit values
    # encoded as described above.

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
        cmap_stack = {l2_cont_id: stack[nonzero_mask]}
    else:
        cmap_stack = fastremap.inverse_component_map(vol_l2.flatten(), stack)
        l2ids = np.array(list(cmap_stack.keys()))
        l2ids = l2ids[l2ids != 0]

    # Initiliaze PCA
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
        l2_max_scaled_coords.append(coords[np.argmax(dts * dist_weight(cv, coords))])
        l2_bboxs.append([np.min(coords, axis=0), np.max(coords, axis=0)])
        l2_sizes.append(len(idxs))
        l2_max_dts.append(dts[max_idx])
        l2_mean_dts.append(np.mean(dts))
        l2_chunk_intersects.append(
            [np.sum(coords == 0, axis=0), np.sum((coords - vol_l2.shape) == 0, axis=0)]
        )

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

        pca.fit(coords_p * cv.resolution)
        l2_pca_comps.append(pca.components_)
        l2_pca_vals.append(pca.singular_values_)

    # In a last step we adjust for the chunk offset.
    offset = chunk_coord + np.array(cv.bounds.to_list()[:3])
    l2_sizes = np.array(np.array(l2_sizes) * np.product(cv.resolution))
    l2_max_dts = np.array(l2_max_dts)
    l2_mean_dts = np.array(l2_mean_dts)
    l2_max_coords = np.array((np.array(l2_max_coords) + offset) * cv.resolution)
    l2_max_scaled_coords = np.array(
        (np.array(l2_max_scaled_coords) + offset) * cv.resolution
    )
    l2_bboxs = np.array(l2_bboxs) + offset
    l2_pca_comps = np.array(l2_pca_comps)
    l2_pca_vals = np.array(l2_pca_vals)
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

    x_area = np.product(cv.resolution[[1, 2]])
    y_area = np.product(cv.resolution[[0, 2]])
    z_area = np.product(cv.resolution[[0, 1]])

    x_dict = Counter(dict(zip(u_x, c_x * x_area)))
    y_dict = Counter(dict(zip(u_y, c_y * y_area)))
    z_dict = Counter(dict(zip(u_z, c_z * z_area)))

    area_dict = x_dict + y_dict + z_dict
    areas = np.array([area_dict[l2id] for l2id in l2ids])

    return {
        "l2id": fastremap.remap(l2ids, l2_contiguous_d).astype(attributes.UINT64.type),
        "size_nm3": l2_sizes.astype(attributes.UINT32.type),
        "area_nm2": areas.astype(attributes.UINT32.type),
        "max_dt_nm": l2_max_dts.astype(attributes.UINT16.type),
        "mean_dt_nm": l2_mean_dts.astype(attributes.FLOAT16.type),
        "rep_coord_nm": l2_max_scaled_coords.astype(attributes.UINT64.type),
        "chunk_intersect_count": l2_chunk_intersects.astype(attributes.UINT16.type),
        "pca_comp": l2_pca_comps.astype(attributes.FLOAT16.type),
        "pca_vals": l2_pca_vals.astype(attributes.FLOAT32.type),
    }


def _l2cache_thread(cg, cv, chunk_coord, timestamp, l2id):
    l2chunk = L2ChunkVolume(cg, cv, chunk_coord, timestamp)
    vol_l2, l2_contiguous_d = l2chunk.get_remapped_segmentation(l2id)
    if np.sum(np.array(list(l2_contiguous_d.values())) != 0) == 0:
        return {}
    return calculate_features(cv, l2chunk.coordinates, vol_l2, l2_contiguous_d, l2id)


def run_l2cache(cg, cv, chunk_coord=None, timestamp=None, l2id=None):
    if chunk_coord is None:
        assert l2id is not None
        chunk_coord = cg.get_chunk_coordinates(l2id)
    chunk_coord = np.array(list(chunk_coord), dtype=int)
    return _l2cache_thread(cg, cv, chunk_coord, timestamp, l2id)


def run_l2cache_batch(cg, cv_path, chunk_coords, timestamp=None):
    ret_dicts = []
    for chunk_coord in chunk_coords:
        ret_dicts.append(run_l2cache(cg, cv_path, chunk_coord, timestamp))

    comb_ret_dict = defaultdict(list)
    for ret_dict in ret_dicts:
        for k in ret_dict:
            comb_ret_dict[k].extend(ret_dict[k])
    return comb_ret_dict


def write_to_db(client: BigTableClient, result_d: dict) -> None:
    from . import attributes
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
