"""The grid and its state.

All per-cell state is float64 arrays of shape (height, width). The atmosphere
is a single scalar carbon pool that closes the carbon budget.
"""

import hashlib
from dataclasses import dataclass, fields

import numpy as np

from sim.params import SimParams


@dataclass
class WorldState:
    tick: int
    elevation: np.ndarray  # 0..1, static
    water_capacity: np.ndarray  # mm, static
    soil_water: np.ndarray  # mm
    plant_c: np.ndarray  # g C
    soil_c: np.ndarray  # g C
    atmosphere_c: float  # g C
    temp_anomaly_c: float  # weather AR(1) memory
    raining: bool

    @property
    def shape(self) -> tuple[int, int]:
        return self.plant_c.shape

    def total_carbon(self) -> float:
        return float(self.atmosphere_c + self.plant_c.sum() + self.soil_c.sum())

    def total_water(self) -> float:
        return float(self.soil_water.sum())

    def state_hash(self) -> str:
        """SHA-256 over every state field, in declaration order. Bit-exact."""
        h = hashlib.sha256()
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                h.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
            else:
                h.update(np.float64(value).tobytes())
        return h.hexdigest()


def _smooth(field: np.ndarray, passes: int) -> np.ndarray:
    """Box blur with edge padding; keeps terrain deterministic and dependency-free."""
    out = field
    for _ in range(passes):
        p = np.pad(out, 1, mode="edge")
        out = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0  # fmt: skip
    return out


def init_world(params: SimParams, rng: np.random.Generator) -> WorldState:
    wp, water, plants, soil = params.world, params.water, params.plants, params.soil
    shape = (wp.height, wp.width)

    elevation = _smooth(rng.random(shape), wp.terrain_smoothing_passes)
    lo, hi = elevation.min(), elevation.max()
    elevation = (elevation - lo) / (hi - lo) if hi > lo else np.zeros(shape)

    capacity = water.capacity_mm * (1.0 + water.valley_capacity_bonus * (1.0 - elevation))
    soil_water = capacity * water.initial_fraction

    plant_c = rng.gamma(shape=2.0, scale=plants.initial_mean_c / 2.0, size=shape)
    plant_c[rng.random(shape) < plants.initial_bare_fraction] = 0.0
    soil_c = np.full(shape, soil.initial_c) * rng.uniform(0.8, 1.2, size=shape)

    return WorldState(
        tick=0,
        elevation=elevation,
        water_capacity=capacity,
        soil_water=soil_water,
        plant_c=plant_c,
        soil_c=soil_c,
        atmosphere_c=params.atmosphere_initial_c,
        temp_anomaly_c=0.0,
        raining=False,
    )
