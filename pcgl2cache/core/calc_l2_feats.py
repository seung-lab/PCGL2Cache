import fastremap
from edt import edt
from collections import Counter
from collections import defaultdict

import numpy as np
from sklearn import decomposition

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from kvdbclient import BigTableClient


def get_chunk_l2_seg(cg, cv, chunk_coord, chunk_size, timestamp):
    bbox = np.array(cv.bounds.to_list())
    vol_coord_start = bbox[:3] + chunk_coord
    vol_coord_end = vol_coord_start + chunk_size
    vol = cv[
        vol_coord_start[0] : vol_coord_end[0],
        vol_coord_start[1] : vol_coord_end[1],
        vol_coord_start[2] : vol_coord_end[2],
    ][..., 0]

    sv_ids = fastremap.unique(vol)
    sv_ids = sv_ids[sv_ids != 0]
    if len(sv_ids) == 0:
        return vol.astype(np.uint32), {}

    l2_ids = cg.get_roots(sv_ids, stop_layer=2, time_stamp=timestamp)
    u_l2_ids = fastremap.unique(l2_ids)
    u_cont_ids = np.arange(1, 1 + len(u_l2_ids))
    cont_ids = fastremap.remap(l2_ids, dict(zip(u_l2_ids, u_cont_ids)))
    fastremap.remap(
        vol, dict(zip(sv_ids, cont_ids)), preserve_missing_labels=True, in_place=True
    )
    return vol.astype(np.uint32), dict(zip(u_cont_ids, u_l2_ids))


def get_l2_seg(cg, cv, chunk_coord, chunk_size, l2_ids):
    bbox = np.array(cv.bounds.to_list())
    vol_coord_start = bbox[:3] + chunk_coord
    vol_coord_end = vol_coord_start + chunk_size
    vol = cv[
        vol_coord_start[0] : vol_coord_end[0],
        vol_coord_start[1] : vol_coord_end[1],
        vol_coord_start[2] : vol_coord_end[2],
    ][..., 0]

    sv_ids = cg.get_children(l2_ids, flatten=True)
    sv_ids = fastremap.unique(sv_ids)
    u_l2_ids = fastremap.unique(l2_ids)
    u_cont_ids = np.arange(1, 1 + len(u_l2_ids))
    cont_ids = fastremap.remap(l2_ids, dict(zip(u_l2_ids, u_cont_ids)))
    fastremap.remap(
        vol, dict(zip(sv_ids, cont_ids)), preserve_missing_labels=True, in_place=True
    )
    return vol.astype(np.uint32), dict(zip(u_cont_ids, u_l2_ids))


def dist_weight(cv, coords):
    mean_coord = np.mean(coords, axis=0)
    dists = np.linalg.norm((coords - mean_coord) * cv.resolution, axis=1)
    return 1 - dists / dists.max()


def calculate_rep_coords(cv, chunk_coord, vol_l2, l2_dict, l2_ids=None):
    from . import attributes

    vol_dt = edt(
        vol_l2,
        anisotropy=cv.resolution,
        black_border=False,
        parallel=1,  # number of threads, <= 0 sets to num cpu
    )

    shape = np.array(vol_l2.shape)
    size = np.product(shape)
    stack = ((vol_dt.astype(np.uint64).flatten()) << 32) + np.arange(
        size, dtype=np.uint64
    )

    cmap_stack = fastremap.inverse_component_map(vol_l2.flatten(), stack)
    pca = decomposition.PCA(3)

    l2_max_coords = []
    l2_max_scaled_coords = []
    l2_bboxs = []
    l2_chunk_intersects = []
    l2_max_dts = []
    l2_mean_dts = []
    l2_sizes = []
    if l2_ids is None:
        l2_ids = np.array(list(cmap_stack.keys()))
    l2_ids = l2_ids[l2_ids != 0]
    l2_pca_comps = []
    l2_pca_vals = []
    for l2_id in l2_ids:
        l2_stack = np.array(cmap_stack[l2_id], dtype=np.uint64)
        dts = l2_stack >> 32
        idxs = l2_stack.astype(np.uint32)
        coords = np.array(np.unravel_index(np.array(idxs), vol_l2.shape)).T

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

    ## Area calculations
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
    areas = np.array([area_dict[l2_id] for l2_id in l2_ids])

    return {
        "l2id": fastremap.remap(l2_ids, l2_dict).astype(attributes.UINT64.type),
        "size_nm3": l2_sizes.astype(attributes.UINT32.type),
        "area_nm2": areas.astype(attributes.UINT32.type),
        "max_dt_nm": l2_max_dts.astype(attributes.UINT16.type),
        "mean_dt_nm": l2_mean_dts.astype(attributes.FLOAT16.type),
        "rep_coord_nm": l2_max_scaled_coords.astype(attributes.UINT64.type),
        "chunk_intersect_count": l2_chunk_intersects.astype(attributes.UINT16.type),
        "pca_comp": l2_pca_comps.astype(attributes.FLOAT16.type),
        "pca_vals": l2_pca_vals.astype(attributes.FLOAT32.type),
    }


def download_and_calculate(cg, cv, chunk_coord, chunk_size, timestamp, l2_ids):
    if l2_ids is None:
        vol_l2, l2_dict = get_chunk_l2_seg(cg, cv, chunk_coord, chunk_size, timestamp)
    else:
        vol_l2, l2_dict = get_l2_seg(cg, cv, chunk_coord, chunk_size, l2_ids)
    if np.sum(np.array(list(l2_dict.values())) != 0) == 0:
        return {}
    return calculate_rep_coords(cv, chunk_coord, vol_l2, l2_dict, l2_ids)


def _l2cache_thread(cg, cv, chunk_coord, timestamp, l2_ids):
    chunk_size = cg.chunk_size.astype(np.int)
    chunk_coord = chunk_coord * chunk_size
    return download_and_calculate(cg, cv, chunk_coord, chunk_size, timestamp, l2_ids)


def run_l2cache(cg_table_id, cv_path, chunk_coord=None, timestamp=None, l2_ids=None):
    from datetime import datetime
    from pychunkedgraph.backend.chunkedgraph import ChunkedGraph
    from cloudvolume import CloudVolume

    cg = ChunkedGraph(cg_table_id)
    if chunk_coord is None:
        assert l2_ids is not None and len(l2_ids) > 0
        chunk_coord = cg.get_chunk_coordinates(l2_ids[0])
    chunk_coord = np.array(list(chunk_coord), dtype=int)
    cv = CloudVolume(
        cv_path, bounded=False, fill_missing=True, progress=False, mip=cg.cv.mip
    )
    return _l2cache_thread(cg, cv, chunk_coord, timestamp, l2_ids)


def run_l2cache_batch(table, cv_path, chunk_coords, timestamp=None):
    ret_dicts = []
    for chunk_coord in chunk_coords:
        ret_dicts.append(run_l2cache(table, cv_path, chunk_coord, timestamp))

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
