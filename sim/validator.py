"""Rule-based intent validator.

Each rule returns a list of human-readable violations; an intent is valid only
if every rule returns none. All rules run, so an agent gets complete feedback
in one go. The same validator runs in the agent's `validate` step (on a
scratch copy of the state) and in the engine when the intent is applied, which
is the authoritative check.

Rules, in order:
    authority    the agent may issue this kind of intent
    geometry     coordinates and patch IDs are on the grid; source != target
    network      patches are within reach and actually carry hyphae
    resources    the source holds what is being moved, and can pay the cost
    bounds       fractions and biases within limits, bias changes gradual
    rate         per-agent intents per tick, and a per-target cooldown
"""

from collections.abc import Iterable

import numpy as np

from sim.intents import (
    AnyIntent,
    Disturb,
    RelocateHyphae,
    SetTradeBias,
    ShuttleNutrients,
    Spill,
    sellable,
    shuttle_cost,
)
from sim.params import SimParams
from sim.patches import PatchError, hops, parse, region
from sim.world import WorldState

AUTHORITY: dict[str, frozenset[str]] = {
    "disturb": frozenset({"user", "director"}),
    "spill": frozenset({"user", "director"}),
    "shuttle_nutrients": frozenset({"user", "mycelium"}),
    "relocate_hyphae": frozenset({"user", "mycelium"}),
    "set_trade_bias": frozenset({"user", "mycelium"}),
}


def _target(intent: AnyIntent) -> str | None:
    return getattr(intent, "to_patch", None) or getattr(intent, "patch", None)


class Validator:
    def __init__(self, params: SimParams):
        self.p = params
        self.net = params.network

    def check(
        self, intent: AnyIntent, state: WorldState, history: Iterable[AnyIntent] = ()
    ) -> list[str]:
        v: list[str] = []
        v += self._authority(intent)
        geometry = self._geometry(intent, state)
        v += geometry
        if not geometry:  # later rules need valid patch IDs
            v += self._network(intent, state)
            v += self._resources(intent, state)
        v += self._bounds(intent, state, patches_ok=not geometry)
        v += self._rate(intent, list(history))
        return v

    # --- rules ---------------------------------------------------------------

    def _authority(self, i: AnyIntent) -> list[str]:
        allowed = AUTHORITY.get(i.kind, frozenset())
        if i.agent not in allowed:
            return [f"authority: agent '{i.agent}' may not issue {i.kind} "
                    f"(allowed: {', '.join(sorted(allowed))})"]  # fmt: skip
        return []

    def _geometry(self, i: AnyIntent, s: WorldState) -> list[str]:
        h, w = s.shape
        if isinstance(i, Disturb):
            if not (0 <= i.x0 < i.x1 <= w and 0 <= i.y0 < i.y1 <= h):
                return [f"geometry: rectangle out of bounds for {w}x{h} grid"]
            return []
        if isinstance(i, Spill):
            if not (0 <= i.x < w and 0 <= i.y < h):
                return [f"geometry: spill centre out of bounds for {w}x{h} grid"]
            return []
        out = []
        for pid in (getattr(i, "from_patch", None), _target(i)):
            if pid is None:
                continue
            try:
                parse(pid, s.shape, self.net.patch_size)
            except PatchError as err:
                out.append(f"geometry: {err}")
        if not out and getattr(i, "from_patch", None) == _target(i):
            out.append("geometry: source and target patch are the same")
        return out

    def _patch_fungal_c(self, pid: str, s: WorldState) -> float:
        return float(s.mycorrhiza.c[region(pid, s.shape, self.net.patch_size)].mean())

    def _network(self, i: AnyIntent, s: WorldState) -> list[str]:
        if not isinstance(i, ShuttleNutrients | RelocateHyphae):
            return []
        out = []
        d = hops(i.from_patch, i.to_patch, s.shape, self.net.patch_size)
        limit = 1 if isinstance(i, RelocateHyphae) else self.net.max_hops
        if d > limit:
            out.append(f"network: {i.from_patch}->{i.to_patch} is {d} hops; limit is {limit}")
        ends = (i.from_patch, i.to_patch) if isinstance(i, ShuttleNutrients) else (i.from_patch,)
        for pid in ends:
            fc = self._patch_fungal_c(pid, s)
            if fc < self.net.min_network_c:
                out.append(f"network: patch {pid} has too little hyphae "
                           f"({fc:.2f} g C/cell < {self.net.min_network_c})")  # fmt: skip
        return out

    def _resources(self, i: AnyIntent, s: WorldState) -> list[str]:
        if not isinstance(i, ShuttleNutrients):
            return []
        out = []
        src = region(i.from_patch, s.shape, self.net.patch_size)
        available = float(sellable(s, i.element, self.p)[src].sum())
        if i.amount_g > 0.9 * available:
            out.append(f"resources: {i.from_patch} has {available:.3f} g sellable "
                       f"{i.element.upper()}; asked for {i.amount_g:.3f} (max 90%)")  # fmt: skip
        cost = shuttle_cost(i, s, self.p)
        budget = self.net.max_cost_fraction * float(s.mycorrhiza.c[src].sum())
        if cost > budget:
            out.append(f"resources: transport costs {cost:.2f} g C but {i.from_patch} can "
                       f"spend at most {budget:.2f}")  # fmt: skip
        return out

    def _bounds(self, i: AnyIntent, s: WorldState, patches_ok: bool) -> list[str]:
        if isinstance(i, RelocateHyphae) and i.fraction > self.net.max_relocate_fraction:
            return [f"bounds: relocate fraction {i.fraction} > {self.net.max_relocate_fraction}"]
        if isinstance(i, SetTradeBias):
            lo, hi = self.net.trade_bias_min, self.net.trade_bias_max
            out = []
            if not lo <= i.bias <= hi:
                out.append(f"bounds: trade bias {i.bias} outside [{lo}, {hi}]")
            if patches_ok:
                now = float(np.mean(s.trade_bias[region(i.patch, s.shape, self.net.patch_size)]))
                if abs(i.bias - now) > self.net.trade_bias_max_step + 1e-12:
                    out.append(f"bounds: bias change {now:.2f}->{i.bias:.2f} exceeds step "
                               f"{self.net.trade_bias_max_step}")  # fmt: skip
            return out
        return []

    def _rate(self, i: AnyIntent, history: list[AnyIntent]) -> list[str]:
        out = []
        same_tick = sum(1 for h in history if h.agent == i.agent and h.tick == i.tick)
        if i.agent != "user" and same_tick >= self.net.max_intents_per_tick:
            out.append(f"rate: {i.agent} already has {same_tick} intents at tick {i.tick} "
                       f"(max {self.net.max_intents_per_tick})")  # fmt: skip
        target = _target(i)
        if target and i.agent != "user":
            cooldown = self.net.cooldown_ticks
            recent = [
                h for h in history
                if h.agent == i.agent and h.kind == i.kind and _target(h) == target
                and 0 <= i.tick - h.tick < cooldown
            ]  # fmt: skip
            if recent:
                last = recent[-1].tick
                out.append(f"rate: {i.kind} on {target} is cooling down "
                           f"(last at tick {last}, cooldown {cooldown})")  # fmt: skip
        return out
