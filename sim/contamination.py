"""A persistent organic contaminant, conserved as its own substance.

It suppresses plant growth, microbial activity and insect survival through a
dose-response curve, and is broken down by bacteria and, more slowly, by
saprotrophic fungi (the basis of bioremediation). Unsorbed contaminant moves
with water; see hydrology.py.
"""

import numpy as np

from sim.env import DT, TickContext, moisture_response, q10
from sim.params import ContaminationParams
from sim.transport import saturating
from sim.world import WorldState


def toxicity(contaminant: np.ndarray, p: ContaminationParams) -> np.ndarray:
    """1 in clean soil, 0.5 at EC50, -> 0 at high dose."""
    return 1.0 / (1.0 + contaminant / p.ec50_g)


def degrade(s: WorldState, p: ContaminationParams, ctx: TickContext) -> None:
    degraders = s.bacteria.c + p.sapro_degradation_share * s.saprotrophs.c
    rate = p.degradation_per_day * DT * degraders * q10(ctx.weather.temp_c, 2.0)
    rate = rate * moisture_response(ctx.moisture, 0.6)
    gone = np.minimum(rate * saturating(s.contaminant, p.degradation_half_sat_g), s.contaminant)
    s.contaminant = s.contaminant - gone
    ctx.ledger.outflow("contaminant", "degradation", float(gone.sum()))
