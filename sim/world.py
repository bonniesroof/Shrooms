"""The grid and its state.

Organic matter lives in named pools, each holding C, N and P arrays of shape
(H, W). Mineral N and P, the contaminant, and soil water (3 layers) are
separate fields. The atmosphere is one scalar carbon pool; N and P exchange
with the outside world through booked boundary flows.
"""

import hashlib
from dataclasses import dataclass, fields

import numpy as np

from sim.params import MicrobeParams, SimParams

POOLS = ("plant", "litter", "som", "bacteria", "saprotrophs", "mycorrhiza", "insects")
ELEMENTS = ("c", "n", "p")


@dataclass
class Pool:
    c: np.ndarray
    n: np.ndarray
    p: np.ndarray

    @classmethod
    def from_c(cls, c: np.ndarray, cn: float, cp: float) -> "Pool":
        return cls(c, c / cn, c / cp)

    def take(self, fraction: np.ndarray) -> "Pool":
        """Remove `fraction` of every element; return what was removed."""
        out = Pool(self.c * fraction, self.n * fraction, self.p * fraction)
        self.c = self.c - out.c
        self.n = self.n - out.n
        self.p = self.p - out.p
        return out

    def add(self, other: "Pool") -> None:
        self.c = self.c + other.c
        self.n = self.n + other.n
        self.p = self.p + other.p

    def total(self, element: str) -> float:
        return float(getattr(self, element).sum())


@dataclass
class WorldState:
    tick: int
    elevation: np.ndarray  # m, static
    layer_capacity: np.ndarray  # (3, H, W) mm, static
    water: np.ndarray  # (3, H, W) mm, top layer first
    plant: Pool
    litter: Pool
    som: Pool
    bacteria: Pool
    saprotrophs: Pool
    mycorrhiza: Pool
    insects: Pool
    mineral_n: np.ndarray
    mineral_p: np.ndarray
    contaminant: np.ndarray  # g
    atmosphere_c: float
    temp_anomaly_c: float
    raining: bool

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    def pool(self, name: str) -> Pool:
        return getattr(self, name)

    def land_total(self, element: str) -> float:
        total = sum(self.pool(name).total(element) for name in POOLS)
        if element == "n":
            total += float(self.mineral_n.sum())
        elif element == "p":
            total += float(self.mineral_p.sum())
        return total

    def total_carbon(self) -> float:
        return self.atmosphere_c + self.land_total("c")

    def total_water(self) -> float:
        return float(self.water.sum())

    def moisture(self) -> np.ndarray:
        """Root-zone relative moisture 0..1 (top two layers)."""
        return np.clip(self.water[:2].sum(0) / self.layer_capacity[:2].sum(0), 0.0, 1.0)

    def state_hash(self) -> str:
        """SHA-256 over every field's raw bytes, in declaration order. Bit-exact."""
        h = hashlib.sha256()
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Pool):
                for el in ELEMENTS:
                    h.update(np.ascontiguousarray(getattr(value, el), np.float64).tobytes())
            elif isinstance(value, np.ndarray):
                h.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
            else:
                h.update(np.float64(value).tobytes())
        return h.hexdigest()


def _smooth(field: np.ndarray, passes: int) -> np.ndarray:
    out = field
    for _ in range(passes):
        p = np.pad(out, 1, mode="edge")
        out = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0  # fmt: skip
    return out


def _guild(shape, m: MicrobeParams, rng: np.random.Generator) -> Pool:
    return Pool.from_c(m.initial_c * rng.uniform(0.7, 1.3, size=shape), m.cn, m.cp)


def _hotspots(shape, params: SimParams, rng: np.random.Generator) -> np.ndarray:
    cp = params.contamination
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    field = np.zeros(shape)
    for _ in range(cp.hotspots):
        cy, cx = rng.uniform(0.15, 0.85) * h, rng.uniform(0.15, 0.85) * w
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        field += cp.hotspot_mass_g * np.exp(-r2 / (2 * cp.hotspot_radius**2))
    return field


def init_world(params: SimParams, rng: np.random.Generator) -> WorldState:
    wp, hp, pp, dp = params.world, params.hydrology, params.plants, params.decomposers
    shape = (wp.height, wp.width)

    elevation = _smooth(rng.random(shape), wp.terrain_smoothing_passes)
    lo, hi = elevation.min(), elevation.max()
    elevation = (elevation - lo) / (hi - lo) * wp.relief if hi > lo else np.zeros(shape)

    capacity = np.stack([np.full(shape, c) for c in hp.layer_capacity_mm])
    water = capacity * hp.initial_fraction

    plant_c = rng.gamma(shape=2.0, scale=pp.initial_mean_c / 2.0, size=shape)
    plant_c[rng.random(shape) < pp.initial_bare_fraction] = 0.0
    litter_c = dp.initial_litter_c * rng.uniform(0.8, 1.2, size=shape)
    som_c = dp.initial_som_c * rng.uniform(0.8, 1.2, size=shape)

    return WorldState(
        tick=0,
        elevation=elevation,
        layer_capacity=capacity,
        water=water,
        plant=Pool.from_c(plant_c, pp.target_cn, pp.target_cp),
        litter=Pool.from_c(
            litter_c, pp.target_cn / (1 - pp.resorption), pp.target_cp / (1 - pp.resorption)
        ),
        som=Pool.from_c(som_c, dp.som_cn, dp.som_cp),
        bacteria=_guild(shape, dp.bacteria, rng),
        saprotrophs=_guild(shape, dp.saprotrophs, rng),
        mycorrhiza=_guild(shape, params.mycorrhiza.guild, rng),
        insects=Pool.from_c(
            np.full(shape, params.insects.initial_c), params.insects.cn, params.insects.cp
        ),
        mineral_n=np.full(shape, params.nutrients.initial_mineral_n),
        mineral_p=np.full(shape, params.nutrients.initial_mineral_p),
        contaminant=_hotspots(shape, params, rng),
        atmosphere_c=params.atmosphere_initial_c,
        temp_anomaly_c=0.0,
        raining=False,
    )
