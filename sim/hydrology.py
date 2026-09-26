"""Layered soil hydrology with lateral flow and solute transport.

Three layers per cell (top, middle, deep). Each tick:

    1. rain enters the top layer
    2. top-layer overflow runs off downslope (one hop); in pits it sinks instead
    3. overflow cascades down the column; bottom overflow drains out of the plot
    4. evapotranspiration draws on layers by root distribution
    5. percolation moves water above half capacity down one layer
    6. subsurface flow moves water downslope in every layer (edges are no-flux)
    7. slow deep drainage leaves the bottom layer

Mobile solutes (dissolved mineral N and P, unsorbed contaminant) move laterally
with subsurface flow and leach out with drainage, in proportion to the water
that carries them. Solutes are tracked per cell, not per layer.
"""

from dataclasses import dataclass

import numpy as np

from sim import TICKS_PER_DAY
from sim.ledger import Ledger
from sim.params import SimParams
from sim.transport import downslope_drops, route
from sim.weather import Weather


@dataclass
class Solutes:
    mineral_n: np.ndarray
    mineral_p: np.ndarray
    contaminant: np.ndarray


def potential_et(weather: Weather, pet_per_wm2: float) -> float:
    """Potential ET (mm/h) from radiation, scaled by temperature (zero below 0 C)."""
    return pet_per_wm2 * weather.shortwave * max(0.0, weather.temp_c) / 20.0


class Hydrology:
    def __init__(self, elevation: np.ndarray, params: SimParams):
        self.p = params.hydrology
        self.nutrients = params.nutrients
        self.sorbed = params.contamination.sorbed_fraction
        drops = downslope_drops(elevation)  # (4, H, W) metres
        total = drops.sum(axis=0)
        # Runoff goes entirely downhill, split by drop; pits keep theirs.
        self.runoff_fractions = np.where(total > 0, drops / np.maximum(total, 1e-12), 0.0)
        self.has_outlet = total > 0
        rate = self.p.lateral_per_day / TICKS_PER_DAY
        per_dir = rate * drops
        cap = np.minimum(per_dir.sum(axis=0), 0.5)  # never send more than half per tick
        self.lateral_fractions = per_dir * np.where(
            per_dir.sum(axis=0) > 0, cap / np.maximum(per_dir.sum(axis=0), 1e-12), 0.0
        )
        self.roots = np.asarray(self.p.root_fraction)[:, None, None]

    def step(
        self,
        water: np.ndarray,
        capacity: np.ndarray,
        canopy_cover: np.ndarray,
        solutes: Solutes,
        weather: Weather,
        ledger: Ledger,
    ) -> tuple[np.ndarray, Solutes]:
        p = self.p
        w = water.copy()

        # 1. Rain.
        w[0] += weather.rain_mm
        ledger.inflow("water", "rain", weather.rain_mm * w[0].size)

        # 2. Runoff downslope from saturated topsoil.
        excess = np.maximum(w[0] - capacity[0], 0.0)
        runoff = np.where(self.has_outlet, excess, 0.0)
        # Outlet cells send all their runoff, so route() returns only what arrives.
        w[0] = w[0] - runoff + route(runoff, self.runoff_fractions)
        ledger.move("water", "runoff", float(runoff.sum()))

        # 3. Overflow cascades down; bottom overflow leaves the plot.
        drained = np.zeros_like(w[0])
        for layer in range(3):
            over = np.maximum(w[layer] - capacity[layer], 0.0)
            w[layer] -= over
            if layer < 2:
                w[layer + 1] += over
            else:
                drained += over

        # 4. Evapotranspiration.
        pet = potential_et(weather, p.pet_per_wm2)
        transp = pet * canopy_cover
        evap = pet * p.bare_soil_evap_fraction * (1.0 - canopy_cover)
        dryness = np.clip(w / capacity, 0.0, 1.0)
        demand = transp * self.roots * dryness
        demand[0] += evap * dryness[0]
        et = np.minimum(demand, w)
        w -= et
        ledger.outflow("water", "evapotranspiration", float(et.sum()))

        # 5. Percolation.
        rate = p.percolation_per_day / TICKS_PER_DAY
        for layer in range(2):
            move = rate * np.maximum(w[layer] - 0.5 * capacity[layer], 0.0)
            w[layer] -= move
            w[layer + 1] += move

        # 6. Subsurface lateral flow, carrying mobile solutes.
        column_before = np.maximum(w.sum(axis=0), 1e-12)
        layer_fractions = self.lateral_fractions[None] * dryness[:, None]  # (3, 4, H, W)
        solute_fractions = (layer_fractions * (w / column_before)[:, None]).sum(axis=0)
        w = route(w, layer_fractions)
        mobile = self._mobile(solutes)
        routed = route(np.stack(list(mobile.values())), solute_fractions)
        moved = dict(zip(mobile, routed, strict=True))

        # 7. Deep drainage.
        deep = w[2] * (p.deep_drainage_per_day / TICKS_PER_DAY)
        w[2] -= deep
        drained += deep
        ledger.outflow("water", "drainage", float(drained.sum()))

        # Leach mobile solutes with the water that left the column.
        leach_fraction = np.clip(drained / (w.sum(axis=0) + drained + 1e-12), 0.0, 1.0)
        out = {}
        for name, mobile in self._mobile(solutes).items():
            fixed = getattr(solutes, name) - mobile
            leached = moved[name] * leach_fraction
            substance = {"mineral_n": "n", "mineral_p": "p", "contaminant": "contaminant"}[name]
            ledger.outflow(substance, "leaching", float(leached.sum()))
            out[name] = fixed + moved[name] - leached
        return w, Solutes(**out)

    def _mobile(self, s: Solutes) -> dict[str, np.ndarray]:
        return {
            "mineral_n": s.mineral_n * self.nutrients.mobile_n_fraction,
            "mineral_p": s.mineral_p * self.nutrients.mobile_p_fraction,
            "contaminant": s.contaminant * (1.0 - self.sorbed),
        }
