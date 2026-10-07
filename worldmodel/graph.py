"""The ecosystem as a typed graph.

Nodes
    patch        one per 8x8 m patch (64 on the default grid), stable IDs "r{row}c{col}"
    atmosphere   one global node: season and recent weather, linked to every patch

Edges (patch -> patch, both directions unless noted)
    adjacent     4-neighbour patches; also the candidate set for fungal links
    downslope    higher -> lower mean elevation, weighted by the drop (water and solutes)
    hyphal       adjacent patches that are both on the fungal network

A snapshot stores only node features; edges are derived from them (elevation
gives downslope, fungal carbon gives hyphal), so datasets stay small and a
graph can be rebuilt from any stored snapshot.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from sim import TICKS_PER_DAY
from sim.engine import Simulation
from sim.patches import patch_means, patch_sums

PATCH_FEATURES = (
    "plant_c", "plant_cn", "litter_c", "som_c", "bacteria_c", "saprotrophs_c",
    "mycorrhiza_c", "insects_c", "mineral_n", "mineral_p", "contaminant",
    "moisture", "elevation", "trade_bias",
)  # fmt: skip
GLOBAL_FEATURES = (
    "doy_sin", "doy_cos", "temp_24h", "temp_7d", "rain_7d_mm", "shortwave_24h",
    "degradability",
)  # fmt: skip
EDGE_TYPES = ("adjacent", "downslope", "hyphal")
NETWORK_MIN_C = 2.0  # mean fungal g C per cell for a patch to be on the network
F = {name: i for i, name in enumerate(PATCH_FEATURES)}


@lru_cache(maxsize=4)
def adjacent_pairs(rows: int, cols: int) -> np.ndarray:
    """(E, 2) undirected neighbour pairs, i < j, in a fixed order (the link-head order)."""
    pairs = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                pairs.append((i, i + 1))
            if r + 1 < rows:
                pairs.append((i, i + cols))
    return np.array(pairs, dtype=np.int64)


@dataclass
class Snapshot:
    """Node features at one moment. `x` is (rows*cols, len(PATCH_FEATURES)), raw units."""

    tick: int
    x: np.ndarray
    g: np.ndarray
    rows: int
    cols: int


def snapshot(sim: Simulation) -> Snapshot:
    s, p = sim.state, sim.params
    size = p.network.patch_size
    rows, cols = s.shape[0] // size, s.shape[1] // size
    plant_c, plant_n = patch_sums(s.plant.c, size), patch_sums(s.plant.n, size)
    cols_ = {
        "plant_c": patch_means(s.plant.c, size),
        "plant_cn": np.where(
            plant_c > 1e-6, plant_c / np.maximum(plant_n, 1e-12), p.plants.target_cn
        ),  # fmt: skip
        "litter_c": patch_means(s.litter.c, size),
        "som_c": patch_means(s.som.c, size),
        "bacteria_c": patch_means(s.bacteria.c, size),
        "saprotrophs_c": patch_means(s.saprotrophs.c, size),
        "mycorrhiza_c": patch_means(s.mycorrhiza.c, size),
        "insects_c": patch_means(s.insects.c, size),
        "mineral_n": patch_means(s.mineral_n, size),
        "mineral_p": patch_means(s.mineral_p, size),
        "contaminant": patch_means(s.contaminant, size),
        "moisture": patch_means(s.moisture(), size),
        "elevation": patch_means(s.elevation, size),
        "trade_bias": patch_means(s.trade_bias, size),
    }
    x = np.stack([cols_[k].ravel() for k in PATCH_FEATURES], axis=1)

    h = sim.history
    day = 1 + (s.tick // TICKS_PER_DAY) % 365
    angle = 2 * np.pi * day / 365

    def recent(key: str, n: int, how=np.mean) -> float:
        vals = h[key][-n:]
        return float(how(vals)) if vals else 0.0

    g = np.array([
        np.sin(angle), np.cos(angle),
        recent("temp_c", TICKS_PER_DAY), recent("temp_c", 7 * TICKS_PER_DAY),
        recent("rain_mm", 7 * TICKS_PER_DAY, np.sum), recent("shortwave", TICKS_PER_DAY),
        p.contamination.degradation_per_day,
    ])  # fmt: skip
    return Snapshot(s.tick, x, g, rows, cols)


def on_network(x: np.ndarray) -> np.ndarray:
    return x[..., F["mycorrhiza_c"]] >= NETWORK_MIN_C


def links(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """(..., E) whether each adjacent pair is a fungal link (both ends on the network)."""
    pairs = adjacent_pairs(rows, cols)
    on = on_network(x)
    return on[..., pairs[:, 0]] & on[..., pairs[:, 1]]


def edges(x: np.ndarray, rows: int, cols: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Typed edges for one snapshot: {type: (index (2, E) source->target, weight (E,))}."""
    pairs = adjacent_pairs(rows, cols)
    both = np.concatenate([pairs, pairs[:, ::-1]])  # directed, both ways
    out = {"adjacent": (both.T, np.ones(len(both)))}
    elev = x[:, F["elevation"]]
    drop = elev[both[:, 0]] - elev[both[:, 1]]
    down = drop > 0
    out["downslope"] = (both[down].T, drop[down])
    link = links(x, rows, cols)
    hy = np.concatenate([pairs[link], pairs[link][:, ::-1]])
    out["hyphal"] = (hy.T.reshape(2, -1), np.ones(len(hy)))
    return out


def dense_adjacency(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """(len(EDGE_TYPES), N, N) row-normalized adjacency matrices (target <- source)."""
    n = rows * cols
    mats = np.zeros((len(EDGE_TYPES), n, n))
    for k, kind in enumerate(EDGE_TYPES):
        (src, dst), w = edges(x, rows, cols)[kind]
        mats[k, dst, src] = w
        deg = mats[k].sum(axis=1, keepdims=True)
        mats[k] = np.divide(mats[k], deg, out=np.zeros_like(mats[k]), where=deg > 0)
    return mats
