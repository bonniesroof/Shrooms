"""Fixed square patches with stable IDs ("r{row}c{col}"), the unit agents act on.

8x8 cells by default, matching the Phase 4 graph builder's patches.
"""

import re
from functools import lru_cache

import numpy as np

PATCH_ID = re.compile(r"^r(\d+)c(\d+)$")


class PatchError(ValueError):
    pass


@lru_cache(maxsize=8)
def patch_grid(shape: tuple[int, int], size: int) -> tuple[int, int]:
    h, w = shape
    if h % size or w % size:
        raise PatchError(f"grid {w}x{h} is not divisible into {size}x{size} patches")
    return h // size, w // size


def parse(patch_id: str, shape: tuple[int, int], size: int) -> tuple[int, int]:
    m = PATCH_ID.match(patch_id)
    if not m:
        raise PatchError(f"bad patch id {patch_id!r}; expected like 'r2c5'")
    r, c = int(m.group(1)), int(m.group(2))
    rows, cols = patch_grid(shape, size)
    if not (0 <= r < rows and 0 <= c < cols):
        raise PatchError(f"patch {patch_id} outside the {rows}x{cols} patch grid")
    return r, c


def region(patch_id: str, shape: tuple[int, int], size: int) -> tuple[slice, slice]:
    r, c = parse(patch_id, shape, size)
    return slice(r * size, (r + 1) * size), slice(c * size, (c + 1) * size)


def hops(a: str, b: str, shape: tuple[int, int], size: int) -> int:
    """Manhattan distance between patches, in patches."""
    (ra, ca), (rb, cb) = parse(a, shape, size), parse(b, shape, size)
    return abs(ra - rb) + abs(ca - cb)


def all_ids(shape: tuple[int, int], size: int) -> list[str]:
    rows, cols = patch_grid(shape, size)
    return [f"r{r}c{c}" for r in range(rows) for c in range(cols)]


def patch_means(field: np.ndarray, size: int) -> np.ndarray:
    """(rows, cols) mean of a cell field over each patch."""
    h, w = field.shape
    return field.reshape(h // size, size, w // size, size).mean(axis=(1, 3))


def patch_sums(field: np.ndarray, size: int) -> np.ndarray:
    h, w = field.shape
    return field.reshape(h // size, size, w // size, size).sum(axis=(1, 3))
