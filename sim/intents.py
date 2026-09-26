"""Typed intents: the only way anything outside the tick loop changes the sim.

Phase 0 has one debug intent (`disturb`) so the replay path is exercised end to
end. LLM agents in Phase 2 will submit intents through this same validator.
"""

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sim.world import WorldState


class InvalidIntent(ValueError):
    pass


class _Intent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tick: int = Field(ge=0)


class Disturb(_Intent):
    """Knock down a fraction of plant biomass in a rectangle; it becomes soil litter."""

    kind: Literal["disturb"] = "disturb"
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive
    fraction: float = Field(gt=0.0, le=1.0)


Intent = Annotated[Disturb, Field(discriminator="kind")]


def validate(intent: Disturb, state: WorldState) -> None:
    h, w = state.shape
    if not (0 <= intent.x0 < intent.x1 <= w and 0 <= intent.y0 < intent.y1 <= h):
        raise InvalidIntent(f"rectangle out of bounds for {w}x{h} grid: {intent}")


def apply(intent: Disturb, state: WorldState, ledger) -> None:
    region = (slice(intent.y0, intent.y1), slice(intent.x0, intent.x1))
    moved = state.plant_c[region] * intent.fraction
    state.plant_c[region] -= moved
    state.soil_c[region] += moved
    ledger.book("disturbance", float(np.sum(moved)))
