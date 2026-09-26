"""Typed intents: the only way anything outside the tick loop changes the sim.

Every intent names the agent that issued it and may carry a rationale. Intents
are checked by sim/validator.py both when an agent proposes them and again,
authoritatively, when the engine applies them. Every apply() is conservative:
mass only moves between pools, or leaves through a booked flow.
"""

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sim.ledger import Ledger
from sim.params import SimParams
from sim.patches import hops, region
from sim.world import Pool, WorldState


class _Intent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tick: int = Field(ge=0)
    agent: str = Field("user", min_length=1, max_length=40)
    rationale: str = Field("", max_length=600)


class Disturb(_Intent):
    """Knock down a fraction of plant biomass in a rectangle; it becomes litter."""

    kind: Literal["disturb"] = "disturb"
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive
    fraction: float = Field(gt=0.0, le=1.0)


class Spill(_Intent):
    """Release contaminant in a Gaussian blob centred on (x, y)."""

    kind: Literal["spill"] = "spill"
    x: float
    y: float
    radius: float = Field(gt=0.0, le=64.0)
    mass_g: float = Field(gt=0.0, le=1e6)


class ShuttleNutrients(_Intent):
    """Move fungal N or P stock through the mycelial network between patches.

    Taken from the source patch's sellable fungal store, delivered into the
    destination's hyphae (where the local market can sell it to plants).
    Costs fungal carbon, respired, in proportion to amount x hops.
    """

    kind: Literal["shuttle_nutrients"] = "shuttle_nutrients"
    from_patch: str
    to_patch: str
    element: Literal["n", "p"]
    amount_g: float = Field(gt=0.0, le=1e4)


class RelocateHyphae(_Intent):
    """Grow into an adjacent patch by moving a fraction of fungal biomass there."""

    kind: Literal["relocate_hyphae"] = "relocate_hyphae"
    from_patch: str
    to_patch: str
    fraction: float = Field(gt=0.0, le=1.0)


class SetTradeBias(_Intent):
    """Set the network's price posture in a patch: < 1 invests in plants, > 1 extracts."""

    kind: Literal["set_trade_bias"] = "set_trade_bias"
    patch: str
    bias: float = Field(gt=0.0, le=10.0)


AnyIntent = Disturb | Spill | ShuttleNutrients | RelocateHyphae | SetTradeBias
Intent = Annotated[AnyIntent, Field(discriminator="kind")]
NETWORK_KINDS = ("shuttle_nutrients", "relocate_hyphae", "set_trade_bias")


def sellable(state: WorldState, element: str, params: SimParams) -> np.ndarray:
    """Fungal store per cell beyond structure and the market reserve."""
    guild = params.mycorrhiza.guild
    ratio = guild.cn if element == "n" else guild.cp
    f = state.mycorrhiza
    store = np.maximum(getattr(f, element) - f.c / ratio, 0.0)
    return store * (1.0 - params.mycorrhiza.reserve_fraction)


def shuttle_cost(intent: ShuttleNutrients, state: WorldState, params: SimParams) -> float:
    net = params.network
    per_g = net.shuttle_cost_c_per_g_n if intent.element == "n" else net.shuttle_cost_c_per_g_p
    distance = hops(intent.from_patch, intent.to_patch, state.shape, net.patch_size)
    return intent.amount_g * per_g * distance


def _respire(state: WorldState, ledger: Ledger, amount: float) -> None:
    state.atmosphere_c += amount
    ledger.outflow("c", "resp_network_transport", amount)


def _share(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    return weights / total if total > 0 else np.full(weights.shape, 1.0 / weights.size)


def apply(intent: AnyIntent, state: WorldState, ledger: Ledger, params: SimParams) -> None:
    size = params.network.patch_size
    f = state.mycorrhiza
    if isinstance(intent, Disturb):
        mask = np.zeros(state.shape)
        mask[intent.y0 : intent.y1, intent.x0 : intent.x1] = intent.fraction
        knocked = state.plant.take(mask)
        state.litter.add(knocked)
        ledger.move("c", "disturbance", float(knocked.c.sum()))

    elif isinstance(intent, Spill):
        h, w = state.shape
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-((yy - intent.y) ** 2 + (xx - intent.x) ** 2) / (2 * intent.radius**2))
        added = intent.mass_g * blob / blob.sum()
        state.contaminant = state.contaminant + added
        ledger.inflow("contaminant", "spill", float(added.sum()))

    elif isinstance(intent, ShuttleNutrients):
        src, dst = (
            region(intent.from_patch, state.shape, size),
            region(intent.to_patch, state.shape, size),
        )
        el = intent.element
        store = sellable(state, el, params)[src]
        taken = intent.amount_g * _share(store)
        field = getattr(f, el).copy()
        field[src] -= taken
        field[dst] += intent.amount_g * _share(f.c[dst])
        setattr(f, el, field)
        ledger.move(el, "network_shuttle", intent.amount_g)
        cost = shuttle_cost(intent, state, params)
        c = f.c.copy()
        c[src] -= cost * _share(f.c[src])
        f.c = c
        _respire(state, ledger, cost)

    elif isinstance(intent, RelocateHyphae):
        src = region(intent.from_patch, state.shape, size)
        dst = region(intent.to_patch, state.shape, size)
        mask = np.zeros(state.shape)
        mask[src] = intent.fraction
        moved = f.take(mask)
        cost = moved.c * params.network.relocate_cost_fraction
        _respire(state, ledger, float(cost.sum()))
        # Hyphae colonize toward roots: weight the destination by plant biomass.
        weights = _share(state.plant.c[dst] + 1.0)
        arrival = Pool(np.zeros(state.shape), np.zeros(state.shape), np.zeros(state.shape))
        arrival.c[dst] = (moved.c - cost).sum() * weights
        arrival.n[dst] = moved.n.sum() * weights
        arrival.p[dst] = moved.p.sum() * weights
        f.add(arrival)
        ledger.move("c", "hyphal_relocation", float(moved.c.sum()))

    elif isinstance(intent, SetTradeBias):
        bias = state.trade_bias.copy()
        bias[region(intent.patch, state.shape, size)] = intent.bias
        state.trade_bias = bias
