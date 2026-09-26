"""Soil water bucket per cell.

Rain fills the bucket; evapotranspiration and drainage empty it. Anything above
the cell's capacity drains immediately, plus slow percolation of what's stored.
Lateral flow between cells is Phase 1 (hydrology).
"""

import numpy as np

from sim.ledger import Ledger
from sim.params import WaterParams
from sim.weather import Weather


def potential_et(weather: Weather, p: WaterParams) -> float:
    """Potential ET (mm/h) from radiation, scaled by temperature (zero below 0 C)."""
    temp_factor = max(0.0, weather.temp_c) / 20.0
    return p.pet_per_wm2 * weather.shortwave * temp_factor


def step_water(
    soil_water: np.ndarray,
    capacity: np.ndarray,
    canopy_cover: np.ndarray,
    weather: Weather,
    p: WaterParams,
    ledger: Ledger,
) -> np.ndarray:
    """Update soil water in place order: rain, ET, drainage. Returns new array."""
    rain = np.full_like(soil_water, weather.rain_mm)
    water = soil_water + rain

    # Canopy transpires at full PET; bare soil evaporates a fraction. Dry soil resists.
    demand = potential_et(weather, p) * (
        canopy_cover + p.bare_soil_evap_fraction * (1.0 - canopy_cover)
    )
    dryness = np.clip(water / capacity, 0.0, 1.0)
    et = np.minimum(demand * dryness, water)
    water = water - et

    overflow = np.maximum(water - capacity, 0.0)
    water = water - overflow
    percolation = water * p.percolation_rate
    water = water - percolation
    drainage = overflow + percolation

    ledger.book("rain", float(rain.sum()))
    ledger.book("evapotranspiration", float(et.sum()))
    ledger.book("drainage", float(drainage.sum()))
    return water
