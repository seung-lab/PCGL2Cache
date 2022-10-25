def calculate_features(l2ids, l2cache_id, cv_path):
    import gc
    import numpy as np
    from cloudvolume import CloudVolume
    from kvdbclient import BigTableClient, get_default_client_info
    from kvdbclient.base import Entry
    from kvdbclient.base import EntryKey
    from pcgl2cache.core.attributes import SIZE_NM3
    from pcgl2cache.core.features import run_l2cache
    from pcgl2cache.core.features import write_to_db

    info = get_default_client_info()
    client = BigTableClient(l2cache_id, config=info.CONFIG)
    cv = CloudVolume(cv_path, bounded=False, fill_missing=True, progress=False)
    for _id in l2ids:
        if cv.get_chunk_layer(_id) != 2:
            continue
        result = run_l2cache(cv, l2id=_id)
        if not result:
            entry = Entry(EntryKey(_id), {SIZE_NM3: np.uint64(0)})
            client.write_entries([entry])
        write_to_db(client, result)
        gc.collect()
