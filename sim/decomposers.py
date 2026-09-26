"""Decomposers: bacteria and saprotrophic fungi eat litter; SOM turns over slowly.

Each guild has fixed biomass stoichiometry. When a guild eats litter:

    assimilated C = CUE x eaten C; the rest is respired
    if the litter carries more N (or P) than growth needs -> the surplus is
        mineralized into the soil's mineral pool
    if it carries less -> the guild immobilizes mineral N/P; if that runs out,
        growth is capped and the unusable carbon is respired ("overflow")

Dead microbes become soil organic matter (SOM). SOM slowly releases dissolved
organic matter that bacteria grow on, mineralizing its surplus N and P.
Bacteria also fix atmospheric N when mineral N is scarce (paying for it in
respired litter carbon) and denitrify mineral N when the soil is waterlogged.

Bacteria prefer wet soil and N-rich litter; saprotrophs tolerate drier soil and
win on tough, N-poor litter. The split of litter between them follows the
litter's C:N.
"""

import numpy as np

from sim.env import DT, TickContext, moisture_response, q10
from sim.params import DecomposerParams, MicrobeParams
from sim.transport import fit, saturating
from sim.world import Pool, WorldState


def _fraction(amount: np.ndarray, pool: np.ndarray) -> np.ndarray:
    return np.where(pool > 0, np.clip(amount / np.maximum(pool, 1e-300), 0.0, 1.0), 0.0)


def grow_on(
    guild: Pool, eaten: Pool, m: MicrobeParams, s: WorldState, ctx: TickContext, name: str
) -> None:
    """Assimilate eaten organic matter into a fixed-stoichiometry guild."""
    growth = m.cue * eaten.c
    # Cap growth by what N and P (from the food plus mineral pools) can support.
    growth = np.minimum(growth, (eaten.n + s.mineral_n) * m.cn)
    growth = np.minimum(growth, (eaten.p + s.mineral_p) * m.cp)
    net_n = eaten.n - growth / m.cn  # + mineralized, - immobilized
    net_p = eaten.p - growth / m.cp
    s.mineral_n = s.mineral_n + net_n
    s.mineral_p = s.mineral_p + net_p
    guild.c = guild.c + growth
    guild.n = guild.n + growth / m.cn
    guild.p = guild.p + growth / m.cp
    ctx.respire(f"resp_{name}", eaten.c - growth)
    ctx.move("n", "net_mineralization", np.maximum(net_n, 0.0))
    ctx.move("n", "immobilization", np.maximum(-net_n, 0.0))


def die(guild: Pool, m: MicrobeParams, s: WorldState, ctx: TickContext, name: str) -> None:
    """Density-dependent death, worse where contaminated. Necromass becomes SOM."""
    rate = m.death_per_day * DT * (1.0 + guild.c / m.crowding_c) * (2.0 - ctx.toxicity)
    dead = guild.take(np.minimum(rate, 1.0))
    s.som.add(dead)
    ctx.move("c", f"death_{name}", dead.c)


def step_decomposers(s: WorldState, p: DecomposerParams, ctx: TickContext) -> None:
    temp = ctx.weather.temp_c
    f_t = q10(temp, p.q10) * ctx.toxicity

    # Litter competition: saprotrophs win as litter C:N rises.
    litter_cn = s.litter.c / np.maximum(s.litter.n, 1e-12)
    sapro_share = litter_cn / (litter_cn + p.litter_cn_sapro_pref)
    wants = []
    for guild, m, share in (
        (s.bacteria, p.bacteria, 1.0 - sapro_share),
        (s.saprotrophs, p.saprotrophs, sapro_share),
    ):
        demand = m.uptake_per_day * DT * guild.c * saturating(s.litter.c, m.half_sat_c)
        wants.append(demand * f_t * moisture_response(ctx.moisture, m.moisture_opt) * 2 * share)
    take_b, take_s = fit(s.litter.c, *wants)
    eaten_b = s.litter.take(_fraction(take_b, s.litter.c))
    eaten_s = s.litter.take(_fraction(take_s, s.litter.c))
    grow_on(s.bacteria, eaten_b, p.bacteria, s, ctx, "bacteria")
    grow_on(s.saprotrophs, eaten_s, p.saprotrophs, s, ctx, "saprotrophs")

    # SOM turnover releases dissolved organic matter, which bacteria eat. It is
    # N-rich (C:N ~12), so bacteria growing on it mineralize N rather than lock it up.
    microbes = s.bacteria.c + s.saprotrophs.c
    activity = saturating(microbes, 10.0) * moisture_response(ctx.moisture, 0.6)
    dom = s.som.take(np.minimum(p.som_turnover_per_day * DT * f_t * activity, 1.0))
    ctx.move("c", "som_to_dom", dom.c)
    grow_on(s.bacteria, dom, p.bacteria, s, ctx, "bacteria")

    # Free-living N fixation when mineral N is scarce, paid for in litter carbon.
    starved = 1.0 - saturating(s.mineral_n, 1.0)
    fixed = p.fixation_per_day * DT * s.bacteria.c * starved * f_t
    if p.fixation_c_cost > 0:  # fixers can only spend the litter carbon that exists
        fixed = np.minimum(fixed, s.litter.c / p.fixation_c_cost)
    cost = fixed * p.fixation_c_cost
    s.litter.c = s.litter.c - cost
    s.mineral_n = s.mineral_n + fixed
    ctx.respire("resp_fixation", cost)
    ctx.ledger.inflow("n", "fixation", float(fixed.sum()))

    # Denitrification in waterlogged soil.
    waterlogged = np.clip((ctx.moisture - 0.8) / 0.2, 0.0, 1.0)
    lost = np.minimum(
        p.denitrification_per_day * DT * s.mineral_n * waterlogged * saturating(s.bacteria.c, 30.0)
        * q10(temp, p.q10), s.mineral_n,
    )  # fmt: skip
    s.mineral_n = s.mineral_n - lost
    ctx.ledger.outflow("n", "denitrification", float(lost.sum()))

    die(s.bacteria, p.bacteria, s, ctx, "bacteria")
    die(s.saprotrophs, p.saprotrophs, s, ctx, "saprotrophs")
