"""Typed intents: the only way anything outside the tick loop changes the sim.

LLM agents in Phase 2 will submit intents through this same validator.
"""

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sim.ledger import Ledger
from sim.world import WorldState


class InvalidIntent(ValueError):
    pass


class _Intent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tick: int = Field(ge=0)


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
    mass_g: float = Field(gt=0.0, le=1e6)  # total released


Intent = Annotated[Disturb | Spill, Field(discriminator="kind")]


def validate(intent: Disturb | Spill, state: WorldState) -> None:
    h, w = state.shape
    if isinstance(intent, Disturb):
        if not (0 <= intent.x0 < intent.x1 <= w and 0 <= intent.y0 < intent.y1 <= h):
            raise InvalidIntent(f"rectangle out of bounds for {w}x{h} grid: {intent}")
    elif not (0 <= intent.x < w and 0 <= intent.y < h):
        raise InvalidIntent(f"spill centre out of bounds for {w}x{h} grid: {intent}")


def apply(intent: Disturb | Spill, state: WorldState, ledger: Ledger) -> None:
    if isinstance(intent, Disturb):
        mask = np.zeros(state.shape)
        mask[intent.y0 : intent.y1, intent.x0 : intent.x1] = intent.fraction
        knocked = state.plant.take(mask)
        state.litter.add(knocked)
        ledger.move("c", "disturbance", float(knocked.c.sum()))
    else:
        h, w = state.shape
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-((yy - intent.y) ** 2 + (xx - intent.x) ** 2) / (2 * intent.radius**2))
        added = intent.mass_g * blob / blob.sum()
        state.contaminant = state.contaminant + added
        ledger.inflow("contaminant", "spill", float(added.sum()))
