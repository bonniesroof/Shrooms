"""Insects: herbivores that wander, eat plants, and return nutrients.

They eat plant tissue (C:N ~30) but build bodies at C:N ~5, so they are
nitrogen-limited: most carbon they eat is respired, and leftover N and P are
excreted straight into the mineral pools, a fast nutrient shortcut past
decomposition. Unassimilated food leaves as frass (litter). They stop feeding
in the cold and overwinter in diapause (low mortality), and spread by random dispersal.
"""

import numpy as np

from sim.env import DT, TickContext, q10
from sim.params import InsectParams
from sim.plants import disperse_pool
from sim.transport import saturating
from sim.world import Pool, WorldState


def step_insects(s: WorldState, p: InsectParams, ctx: TickContext) -> None:
    ins, temp = s.insects, ctx.weather.temp_c
    warm = temp >= p.min_feeding_temp_c

    if warm:
        rate = p.feeding_per_day * DT * ins.c * saturating(s.plant.c, p.half_sat_plant_c)
        rate = np.minimum(rate * q10(temp, 2.0), s.plant.c * 0.5)
        frac = np.where(s.plant.c > 0, rate / np.maximum(s.plant.c, 1e-300), 0.0)
        eaten = s.plant.take(frac)
        ctx.move("c", "herbivory", eaten.c)

        food = Pool(eaten.c * p.assimilation, eaten.n * p.nutrient_assimilation,
                    eaten.p * p.nutrient_assimilation)  # fmt: skip
        s.litter.add(Pool(eaten.c - food.c, eaten.n - food.n, eaten.p - food.p))  # frass
        growth = np.minimum(p.growth_efficiency * food.c, food.n * p.cn)
        growth = np.minimum(growth, food.p * p.cp)
        ins.c = ins.c + growth
        ins.n = ins.n + growth / p.cn
        ins.p = ins.p + growth / p.cp
        s.mineral_n = s.mineral_n + food.n - growth / p.cn  # excretion
        s.mineral_p = s.mineral_p + food.p - growth / p.cp
        ctx.respire("resp_insects", food.c - growth)

    death = p.death_per_day if warm else p.diapause_death_per_day
    rate = death * DT * (1.0 + ins.c / p.crowding_c) * (2.0 - ctx.toxicity)
    dead = ins.take(np.minimum(rate, 1.0))
    s.litter.add(dead)
    ctx.move("c", "death_insects", dead.c)

    disperse_pool(ins, p.dispersal_per_day * DT)
