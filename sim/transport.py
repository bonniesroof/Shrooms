"""Conservative lateral transport and shared-resource helpers.

Lateral moves go to the 4 neighbours (N, S, W, E) with no-flux edges: whatever
leaves a cell arrives in a neighbour, so totals change only by float rounding.
"""

from functools import lru_cache

import numba
import numpy as np

# Direction index -> (row offset, col offset)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def edge_mask(shape: tuple[int, int]) -> np.ndarray:
    """(4, H, W) array: 1 where the neighbour in that direction exists."""
    h, w = shape
    m = np.ones((4, h, w))
    m[0, 0, :] = 0
    m[1, -1, :] = 0
    m[2, :, 0] = 0
    m[3, :, -1] = 0
    return m


@lru_cache(maxsize=32)
def uniform_fractions(shape: tuple[int, int], rate: float) -> np.ndarray:
    """Diffusion: each cell sends `rate / 4` of its content to each existing neighbour.

    Edge cells send less in total rather than more per neighbour, so the flux
    between any two cells is symmetric and a uniform field stays uniform.
    Cached and read-only: callers must not modify the returned array.
    """
    out = rate / 4.0 * edge_mask(shape)
    out.flags.writeable = False
    return out


def downslope_drops(elevation: np.ndarray) -> np.ndarray:
    """(4, H, W) elevation drop (m) to each neighbour; 0 uphill or off-grid."""
    p = np.pad(elevation, 1, mode="edge")
    neighbours = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])  # N, S, W, E
    return np.maximum(elevation - neighbours, 0.0) * edge_mask(elevation.shape)


def route_reference(amount: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Pure-numpy route(). Kept as the specification the compiled kernel must match bit for bit."""
    sent = amount[..., None, :, :] * fractions  # (..., 4, H, W)
    out = amount - sent.sum(axis=-3)
    out[..., :-1, :] += sent[..., 0, 1:, :]  # northward: row r -> r-1
    out[..., 1:, :] += sent[..., 1, :-1, :]  # southward
    out[..., :, :-1] += sent[..., 2, :, 1:]  # westward
    out[..., :, 1:] += sent[..., 3, :, :-1]  # eastward
    return out


@numba.njit(cache=True)
def _route_kernel(a, f, out):  # a (K,H,W), f (K,4,H,W) or (1,4,H,W) shared, out (K,H,W)
    k_, h, w = a.shape
    shared = f.shape[0] == 1
    for kk in range(k_):
        k = 0 if shared else kk
        for r in range(h):
            for c in range(w):
                x = a[kk, r, c]
                # Same operations in the same order as route_reference, so the
                # result is bit-identical (no fastmath, no reassociation).
                sent = ((x * f[k, 0, r, c] + x * f[k, 1, r, c]) + x * f[k, 2, r, c]) + x * f[
                    k, 3, r, c
                ]
                v = x - sent
                if r + 1 < h:
                    v += a[kk, r + 1, c] * f[k, 0, r + 1, c]
                if r >= 1:
                    v += a[kk, r - 1, c] * f[k, 1, r - 1, c]
                if c + 1 < w:
                    v += a[kk, r, c + 1] * f[k, 2, r, c + 1]
                if c >= 1:
                    v += a[kk, r, c - 1] * f[k, 3, r, c - 1]
                out[kk, r, c] = v


def route(amount: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Move `amount * fractions[d]` from each cell to its neighbour in direction d.

    `amount` is (..., H, W), e.g. one field or a stack of layers or elements.
    `fractions` is (..., 4, H, W), broadcast against it, with per-cell sums <= 1.
    Returns the new field(s). Compiled with Numba; see route_reference for the spec.
    """
    h, w = amount.shape[-2:]
    lead = amount.shape[:-2]
    a = np.ascontiguousarray(amount, dtype=np.float64).reshape(-1, h, w)
    if fractions.ndim == 3:  # one set of fractions shared by every field: no copy
        f = fractions.reshape(1, 4, h, w)
    else:
        f = np.broadcast_to(fractions, (*lead, 4, h, w)).reshape(-1, 4, h, w)
    out = np.empty_like(a)
    _route_kernel(a, np.ascontiguousarray(f, dtype=np.float64), out)
    return out.reshape(amount.shape)


def fit(available: np.ndarray, *demands: np.ndarray) -> list[np.ndarray]:
    """Scale competing demands down proportionally so their sum never exceeds supply."""
    total = sum(demands)
    scale = np.where(total > available, available / np.maximum(total, 1e-300), 1.0)
    return [d * scale for d in demands]


def saturating(x: np.ndarray, half_sat: float) -> np.ndarray:
    """Michaelis-Menten x / (K + x), safe at x = 0."""
    return x / (half_sat + x)
