"""Plants: photosynthesis, respiration, root nutrient uptake, turnover, disturbance.

Plant stoichiometry is flexible. Growth slows as N:C or P:C falls below target,
so nutrients limit carbon gain. N and P come from root uptake and from
mycorrhizal trade (see mycorrhiza.py). Before litterfall, plants resorb a share
of leaf N and P, so litter is poorer in nutrients than living tissue.
"""

import numpy as np

from sim.env import DT, TickContext, q10
from sim.params import PlantParams
from sim.transport import route, saturating, uniform_fractions
from sim.world import Pool, WorldState


def canopy_cover(plant_c: np.ndarray, p: PlantParams) -> np.ndarray:
    return 1.0 - np.exp(-p.cover_k * plant_c)


def nutrient_ratios(plant: Pool, p: PlantParams) -> tuple[np.ndarray, np.ndarray]:
    """N and P content relative to target (1 = on target). Bare cells count as satisfied."""
    safe_c = np.maximum(plant.c, 1e-9)
    has = plant.c > 1e-9
    rn = np.where(has, plant.n * p.target_cn / safe_c, 1.0)
    rp = np.where(has, plant.p * p.target_cp / safe_c, 1.0)
    return rn, rp


def photosynthesis(s: WorldState, p: PlantParams, ctx: TickContext) -> np.ndarray:
    """Fix carbon from the atmosphere into plants; respire maintenance. Returns GPP."""
    wx = ctx.weather
    rn, rp = nutrient_ratios(s.plant, p)
    f_nut = np.clip(np.minimum(rn, rp), 0.0, 1.0)
    cover = canopy_cover(s.plant.c, p)
    f_temp = np.exp(-(((wx.temp_c - p.temp_opt_c) / p.temp_width_c) ** 2))
    f_water = np.clip(ctx.moisture / p.water_stress_fraction, 0.0, 1.0)
    f_crowd = np.clip(1.0 - s.plant.c / p.max_c, 0.0, 1.0)
    gpp = p.light_use_efficiency * wx.shortwave * cover * f_temp * f_water * f_crowd
    gpp = gpp * f_nut * ctx.toxicity

    resp = np.minimum(s.plant.c * p.respiration_per_day * DT * q10(wx.temp_c, p.q10), s.plant.c)
    s.plant.c = s.plant.c + gpp - resp
    ctx.fix_carbon("gpp", gpp)
    ctx.respire("resp_plant", resp)
    return gpp


def root_uptake(s: WorldState, p: PlantParams, ctx: TickContext) -> None:
    """Roots take mineral N and P from what decomposers and fungi left."""
    rn, rp = nutrient_ratios(s.plant, p)
    f_water = np.clip(ctx.moisture / p.water_stress_fraction, 0.0, 1.0)
    up_n = p.root_uptake_n_per_day * DT * s.plant.c * saturating(s.mineral_n, p.root_half_sat_n)
    up_p = p.root_uptake_p_per_day * DT * s.plant.c * saturating(s.mineral_p, p.root_half_sat_p)
    up_n = np.minimum(up_n * f_water * np.clip(1.2 - rn, 0.0, 1.0), s.mineral_n)
    up_p = np.minimum(up_p * f_water * np.clip(1.2 - rp, 0.0, 1.0), s.mineral_p)
    s.mineral_n = s.mineral_n - up_n
    s.mineral_p = s.mineral_p - up_p
    s.plant.n = s.plant.n + up_n
    s.plant.p = s.plant.p + up_p
    ctx.move("n", "root_uptake", up_n)
    ctx.move("p", "root_uptake", up_p)


def turnover(s: WorldState, p: PlantParams, rng: np.random.Generator, ctx: TickContext) -> None:
    """Litterfall (with nutrient resorption) and random disturbance, both to litter."""
    frac = p.turnover_per_day * DT
    fall = Pool(s.plant.c * frac, s.plant.n * frac * (1 - p.resorption),
                s.plant.p * frac * (1 - p.resorption))  # fmt: skip
    s.plant.c = s.plant.c - fall.c
    s.plant.n = s.plant.n - fall.n
    s.plant.p = s.plant.p - fall.p
    s.litter.add(fall)
    ctx.move("c", "litterfall", fall.c)

    struck = rng.random(s.shape) < p.disturbance_prob
    knocked = s.plant.take(np.where(struck, p.disturbance_severity, 0.0))
    s.litter.add(knocked)
    ctx.move("c", "disturbance", knocked.c)


def disperse_pool(pool: Pool, rate: float) -> None:
    """Spread a share of a pool to its neighbours, all elements in proportion."""
    moved = route(np.stack([pool.c, pool.n, pool.p]), uniform_fractions(pool.c.shape, rate))
    pool.c, pool.n, pool.p = moved[0], moved[1], moved[2]
