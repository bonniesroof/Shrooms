"""Mycorrhizal fungi, model A: a biological market.

The fungi forage mineral N and P far more efficiently than roots (lower
half-saturation), and hold what they gather beyond their own structural needs
as a tradeable store. Each tick, in every cell:

    1. fungi forage N and P from the mineral pools, and mine them directly out of
       soil organic matter with enzymes (roots can do neither as well)
    2. the plant sets a carbon budget in proportion to its nutrient demand
       (how far its N:C and P:C sit below target) and to current GPP
    3. a price is set per nutrient: dearer when plant demand is high and the
       fungal store is thin, cheaper when demand is low and the store is full
    4. fungi hold back a reserve for their own growth and sell the rest; the
       plant pays only for what is delivered, never in advance
    5. fungi grow on the carbon they earn, up to what their own stoichiometry
       allows, and respire the rest

Nothing is hard-wired about who helps whom. A well-fed plant stops buying, and
unpaid fungi decline; a starved plant buys heavily and its fungi flourish.
Models B (source-sink) and C (hybrid) arrive in Phase 6 for comparison.
"""

import numpy as np

from sim.decomposers import die
from sim.env import DT, TickContext, moisture_response, q10
from sim.params import MycorrhizaParams, PlantParams
from sim.plants import nutrient_ratios
from sim.transport import saturating
from sim.world import WorldState


def step_mycorrhiza(
    s: WorldState, gpp: np.ndarray, p: MycorrhizaParams, pp: PlantParams, ctx: TickContext
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run foraging and trade. Returns (nutrients delivered by element, carbon paid)."""
    m, f = p.guild, s.mycorrhiza
    activity = q10(ctx.weather.temp_c, 2.0) * moisture_response(ctx.moisture, m.moisture_opt)
    activity = activity * ctx.toxicity

    # 1. Forage.
    got_n = p.forage_n_per_day * DT * f.c * saturating(s.mineral_n, p.half_sat_n) * activity
    got_p = p.forage_p_per_day * DT * f.c * saturating(s.mineral_p, p.half_sat_p) * activity
    got_n, got_p = np.minimum(got_n, s.mineral_n), np.minimum(got_p, s.mineral_p)
    s.mineral_n, s.mineral_p = s.mineral_n - got_n, s.mineral_p - got_p
    f.n, f.p = f.n + got_n, f.p + got_p
    ctx.move("n", "fungal_foraging", got_n)
    ctx.move("p", "fungal_foraging", got_p)

    # 1b. Mine N and P directly from soil organic matter (roots can't).
    mined_n = p.mine_n_per_day * DT * f.c * saturating(s.som.n, p.mine_half_sat_n) * activity
    mined_p = p.mine_p_per_day * DT * f.c * saturating(s.som.p, p.mine_half_sat_p) * activity
    mined_n, mined_p = np.minimum(mined_n, s.som.n), np.minimum(mined_p, s.som.p)
    s.som.n, s.som.p = s.som.n - mined_n, s.som.p - mined_p
    f.n, f.p = f.n + mined_n, f.p + mined_p
    ctx.move("n", "fungal_som_mining", mined_n)
    ctx.move("p", "fungal_som_mining", mined_p)

    # 2. Plant demand and carbon budget.
    rn, rp = nutrient_ratios(s.plant, pp)
    demand_n, demand_p = np.clip(1.0 - rn, 0.0, 1.0), np.clip(1.0 - rp, 0.0, 1.0)
    total_demand = demand_n + demand_p
    budget = p.max_c_offer_fraction * gpp * np.clip(total_demand, 0.0, 1.0)
    budget = np.minimum(budget, s.plant.c * 0.05)
    share_n = np.where(total_demand > 0, demand_n / np.maximum(total_demand, 1e-12), 0.0)

    # 3-4. Price and trade, per nutrient.
    delivered, paid = {}, np.zeros_like(budget)
    for el, demand, share, base in (
        ("n", demand_n, share_n, p.price_n),
        ("p", demand_p, 1.0 - share_n, p.price_p),
    ):
        ratio = m.cn if el == "n" else m.cp
        structural = f.c / ratio
        store = np.maximum(getattr(f, el) - structural, 0.0)
        stock = store / np.maximum(structural, 1e-12)  # store relative to own needs
        price = base * np.clip(1.0 + p.price_elasticity * (demand - stock), 0.25, 4.0)
        amount = np.minimum(store * (1.0 - p.reserve_fraction), budget * share / price)
        delivered[el] = amount
        paid = paid + amount * price
        setattr(f, el, getattr(f, el) - amount)
        setattr(s.plant, el, getattr(s.plant, el) + amount)
        ctx.move(el, "mycorrhizal_delivery", amount)
    s.plant.c = s.plant.c - paid
    ctx.move("c", "mycorrhizal_payment", paid)

    # 5. Fungal growth on earned carbon, capped by its own N and P.
    growth = m.cue * paid
    growth = np.minimum(growth, np.maximum(f.n * m.cn - f.c, 0.0))
    growth = np.minimum(growth, np.maximum(f.p * m.cp - f.c, 0.0))
    f.c = f.c + growth
    ctx.respire("resp_mycorrhiza", paid - growth)

    die(f, m, s, ctx, "mycorrhiza")
    return delivered, paid
