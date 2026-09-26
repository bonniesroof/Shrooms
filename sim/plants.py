"""Plants and soil carbon.

Carbon pools per cell: plant_c (living biomass) and soil_c (litter + organic
matter). The atmosphere is one scalar pool. Flows:

    atmosphere --gpp--------------> plant
    plant ------autotrophic_resp--> atmosphere
    plant ------litterfall--------> soil
    plant ------disturbance-------> soil
    soil -------heterotrophic_resp> atmosphere
    plant ------dispersal---------> neighbouring plant (internal, conserves C)

Every flow is computed from pre-tick state, capped so no pool goes negative,
then applied as an exact subtract-from-source / add-to-sink pair.
"""

import numpy as np

from sim.ledger import Ledger
from sim.params import PlantParams, SoilParams
from sim.weather import Weather


def canopy_cover(plant_c: np.ndarray, p: PlantParams) -> np.ndarray:
    return 1.0 - np.exp(-p.cover_k * plant_c)


def _q10(temp_c: float, q10: float) -> float:
    return q10 ** ((temp_c - 20.0) / 10.0)


def disperse(plant_c: np.ndarray, rate: float) -> np.ndarray:
    """Each cell exports `rate` of its biomass, split evenly among its 4-neighbours.

    Non-wrapping edges: edge cells split among fewer neighbours, so nothing leaves
    the grid. Conserves carbon up to float rounding.
    """
    h, w = plant_c.shape
    neighbours = np.full((h, w), 4.0)
    neighbours[0, :] -= 1
    neighbours[-1, :] -= 1
    neighbours[:, 0] -= 1
    neighbours[:, -1] -= 1

    share = plant_c * rate / neighbours
    out = plant_c - share * neighbours
    out[1:, :] += share[:-1, :]  # from north
    out[:-1, :] += share[1:, :]  # from south
    out[:, 1:] += share[:, :-1]  # from west
    out[:, :-1] += share[:, 1:]  # from east
    return out


def step_carbon(
    plant_c: np.ndarray,
    soil_c: np.ndarray,
    atmosphere_c: float,
    soil_water: np.ndarray,
    water_capacity: np.ndarray,
    weather: Weather,
    pp: PlantParams,
    sp: SoilParams,
    rng: np.random.Generator,
    ledger: Ledger,
) -> tuple[np.ndarray, np.ndarray, float]:
    moisture = np.clip(soil_water / water_capacity, 0.0, 1.0)

    # Gross primary production: light x cover x temperature x water x crowding.
    cover = canopy_cover(plant_c, pp)
    f_temp = np.exp(-(((weather.temp_c - pp.temp_opt_c) / pp.temp_width_c) ** 2))
    f_water = np.clip(moisture / pp.water_stress_fraction, 0.0, 1.0)
    f_crowd = np.clip(1.0 - plant_c / pp.max_c, 0.0, 1.0)
    gpp = pp.light_use_efficiency * weather.shortwave * cover * f_temp * f_water * f_crowd
    gpp = np.minimum(gpp, atmosphere_c / gpp.size)  # never draw the atmosphere negative

    # Losses from pre-tick plant biomass, capped to what exists.
    ra = plant_c * pp.respiration_rate_20c * _q10(weather.temp_c, pp.q10)
    litter = plant_c * pp.turnover_rate
    struck = rng.random(plant_c.shape) < pp.disturbance_prob
    disturbed = np.where(struck, plant_c * pp.disturbance_severity, 0.0)
    losses = ra + litter + disturbed
    scale = np.where(losses > plant_c, plant_c / np.maximum(losses, 1e-300), 1.0)
    ra, litter, disturbed = ra * scale, litter * scale, disturbed * scale

    # Decomposition: temperature and moisture limited.
    rh = soil_c * sp.decomposition_rate_20c * _q10(weather.temp_c, sp.q10) * moisture
    rh = np.minimum(rh, soil_c)

    plant_c = plant_c + gpp - ra - litter - disturbed
    soil_c = soil_c + litter + disturbed - rh
    atmosphere_c = atmosphere_c - float(gpp.sum()) + float(ra.sum()) + float(rh.sum())
    plant_c = disperse(plant_c, pp.dispersal_rate)

    for name, flow in (
        ("gpp", gpp),
        ("autotrophic_resp", ra),
        ("litterfall", litter),
        ("disturbance", disturbed),
        ("heterotrophic_resp", rh),
    ):
        ledger.book(name, float(flow.sum()))
    return plant_c, soil_c, atmosphere_c
