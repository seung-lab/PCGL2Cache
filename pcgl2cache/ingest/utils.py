from typing import Sequence

chunk_id_str = lambda layer, coords: f"{layer}_{'_'.join(map(str, coords))}"


def chunked(l: Sequence, n: int):
    """Yield successive n-sized chunks from l."""
    if n < 1:
        n = len(l)
    for i in range(0, len(l), n):
        yield l[i : i + n]
