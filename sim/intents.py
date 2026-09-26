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


# --- Remediation tools (the player) and scenario events (the director) --------
# Each acts on a Gaussian blob of the given radius around (x, y), or a rectangle.


class _Blob(_Intent):
    x: float
    y: float
    radius: float = Field(3.0, gt=0.0, le=16.0)


class Inoculate(_Blob):
    """Bioaugmentation: bring in cultured microbes (bacteria, saprotrophs or mycorrhizae)."""

    kind: Literal["inoculate"] = "inoculate"
    guild: Literal["bacteria", "saprotrophs", "mycorrhiza"]
    mass_c_g: float = Field(gt=0.0, le=5000.0)


class Amend(_Blob):
    """Spread compost: N-rich litter that feeds decomposers and holds water."""

    kind: Literal["amend"] = "amend"
    mass_c_g: float = Field(gt=0.0, le=50000.0)


class Seed(_Blob):
    """Sow plants: young biomass at target stoichiometry."""

    kind: Literal["seed"] = "seed"
    mass_c_g: float = Field(gt=0.0, le=20000.0)


class Irrigate(_Blob):
    """Add water to the topsoil."""

    kind: Literal["irrigate"] = "irrigate"
    water_mm: float = Field(gt=0.0, le=100.0)  # at the blob centre


class Excavate(_Intent):
    """Dig out and haul away a share of the soil in a rectangle, contaminant and all."""

    kind: Literal["excavate"] = "excavate"
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive
    fraction: float = Field(gt=0.0, le=0.9)


class PestOutbreak(_Blob):
    """Director event: a swarm of herbivorous insects arrives."""

    kind: Literal["pest_outbreak"] = "pest_outbreak"
    mass_c_g: float = Field(gt=0.0, le=500.0)


class Downpour(_Blob):
    """Director event: a cloudburst over part of the plot."""

    kind: Literal["downpour"] = "downpour"
    water_mm: float = Field(gt=0.0, le=150.0)  # at the blob centre


TOOL_KINDS = ("inoculate", "amend", "seed", "irrigate", "excavate")
DIRECTOR_KINDS = ("pest_outbreak", "downpour")

AnyIntent = (
    Disturb | Spill | ShuttleNutrients | RelocateHyphae | SetTradeBias
    | Inoculate | Amend | Seed | Irrigate | Excavate | PestOutbreak | Downpour
)  # fmt: skip
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


def blob(shape: tuple[int, int], x: float, y: float, radius: float) -> np.ndarray:
    """Gaussian weights around (x, y) that sum to 1."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * radius**2))
    return g / g.sum()


def peak_blob(shape: tuple[int, int], x: float, y: float, radius: float) -> np.ndarray:
    """Gaussian weights around (x, y) that equal 1 at the centre."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * radius**2))


COMPOST_CN, COMPOST_CP = 15.0, 120.0


def _import_pool(state, ledger, pool, c, cn, cp, flow) -> None:
    add = Pool(c, c / cn, c / cp)
    pool.add(add)
    ledger.inflow("c", flow, float(add.c.sum()), external=True)
    ledger.inflow("n", flow, float(add.n.sum()))
    ledger.inflow("p", flow, float(add.p.sum()))


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
        added = intent.mass_g * blob(state.shape, intent.x, intent.y, intent.radius)
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

    elif isinstance(intent, Inoculate):
        w = blob(state.shape, intent.x, intent.y, intent.radius)
        guild = {
            "bacteria": params.decomposers.bacteria,
            "saprotrophs": params.decomposers.saprotrophs,
            "mycorrhiza": params.mycorrhiza.guild,
        }[intent.guild]
        _import_pool(state, ledger, state.pool(intent.guild), intent.mass_c_g * w,
                     guild.cn, guild.cp, "inoculation")  # fmt: skip

    elif isinstance(intent, Amend):
        w = blob(state.shape, intent.x, intent.y, intent.radius)
        _import_pool(state, ledger, state.litter, intent.mass_c_g * w,
                     COMPOST_CN, COMPOST_CP, "compost")  # fmt: skip

    elif isinstance(intent, Seed):
        w = blob(state.shape, intent.x, intent.y, intent.radius)
        pp = params.plants
        _import_pool(state, ledger, state.plant, intent.mass_c_g * w,
                     pp.target_cn, pp.target_cp, "seeding")  # fmt: skip

    elif isinstance(intent, PestOutbreak):
        w = blob(state.shape, intent.x, intent.y, intent.radius)
        ip = params.insects
        _import_pool(state, ledger, state.insects, intent.mass_c_g * w,
                     ip.cn, ip.cp, "migration")  # fmt: skip

    elif isinstance(intent, Irrigate | Downpour):
        added = intent.water_mm * peak_blob(state.shape, intent.x, intent.y, intent.radius)
        water = state.water.copy()
        water[0] += added
        state.water = water
        flow = "irrigation" if isinstance(intent, Irrigate) else "storm"
        ledger.inflow("water", flow, float(added.sum()))

    elif isinstance(intent, Excavate):
        mask = np.zeros(state.shape)
        mask[intent.y0 : intent.y1, intent.x0 : intent.x1] = intent.fraction
        removed = {"c": 0.0, "n": 0.0, "p": 0.0}
        for name in ("plant", "litter", "som", "bacteria", "saprotrophs", "mycorrhiza", "insects"):
            gone = state.pool(name).take(mask)
            for el in removed:
                removed[el] += float(getattr(gone, el).sum())
        dug_n, dug_p = state.mineral_n * mask, state.mineral_p * mask
        state.mineral_n, state.mineral_p = state.mineral_n - dug_n, state.mineral_p - dug_p
        removed["n"] += float(dug_n.sum())
        removed["p"] += float(dug_p.sum())
        dug = state.contaminant * mask
        state.contaminant = state.contaminant - dug
        ledger.outflow("c", "excavation", removed["c"], external=True)
        ledger.outflow("n", "excavation", removed["n"])
        ledger.outflow("p", "excavation", removed["p"])
        ledger.outflow("contaminant", "excavation", float(dug.sum()))

    elif isinstance(intent, SetTradeBias):
        bias = state.trade_bias.copy()
        bias[region(intent.patch, state.shape, size)] = intent.bias
        state.trade_bias = bias
