"""The tick loop.

One tick = one sim-hour. Order within a tick is fixed:

    1. apply accepted intents scheduled for this tick
    2. weather
    3. hydrology (water, plus dissolved N, P and contaminant)
    4. plants photosynthesize and respire
    5. decomposers eat litter, turn over SOM, fix and denitrify N
    6. contaminant degradation
    7. mycorrhizal foraging and trade (model A market)
    8. roots take up what nutrients remain
    9. plant litterfall and disturbance
   10. insects feed, excrete, die and disperse
   11. plant and microbial dispersal; N deposition and P weathering
   12. mass-balance check for C, N, P, water and contaminant

The order sets who gets first claim on shared pools: microbes before fungi
before roots, as in real soils where microbes out-compete plants short term.
"""

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from sim import SIM_VERSION
from sim.contamination import degrade, toxicity
from sim.decomposers import step_decomposers
from sim.env import DT, TickContext
from sim.events import EventDetector, EventLog, date_label
from sim.hydrology import Hydrology, Solutes
from sim.insects import step_insects
from sim.intents import AnyIntent, apply
from sim.ledger import Ledger
from sim.mycorrhiza import step_mycorrhiza
from sim.params import SimParams
from sim.plants import canopy_cover, disperse_pool, photosynthesis, root_uptake, turnover
from sim.rng import make_streams
from sim.validator import Validator
from sim.weather import Weather, step_weather
from sim.world import POOLS, WorldState, init_world

log = logging.getLogger("shrooms.sim")

HISTORY_FIELDS = (
    "temp_c", "shortwave", "rain_mm", "atmosphere_c", "soil_water", "canopy_cover",
    "mineral_n", "mineral_p", "contaminant", "gpp", "trade_c", "trade_n", "trade_p",
    *(f"{pool}_c" for pool in POOLS),
)  # fmt: skip


@dataclass
class Simulation:
    seed: int
    params: SimParams = field(default_factory=SimParams)
    check_balance: bool = True

    def __post_init__(self) -> None:
        self.rng = make_streams(self.seed)
        self.state: WorldState = init_world(self.params, self.rng["init"])
        self.hydrology = Hydrology(self.state.elevation, self.params)
        self.ledger = Ledger(self.totals(), self.state.atmosphere_c)
        self.validator = Validator(self.params)
        self.events = EventLog()
        self.detector = EventDetector(float(self.state.contaminant.sum()), self.state.plant.c.size)
        self.pending: dict[int, list[AnyIntent]] = defaultdict(list)
        self.accepted: list[AnyIntent] = []
        self.rejected: list[tuple[AnyIntent, str]] = []
        self.history: dict[str, list[float]] = {k: [] for k in HISTORY_FIELDS}
        self.last_weather: Weather | None = None
        self.version = SIM_VERSION

    def totals(self) -> dict[str, float]:
        s = self.state
        return {
            "c": s.land_total("c"),
            "n": s.land_total("n"),
            "p": s.land_total("p"),
            "water": s.total_water(),
            "contaminant": float(s.contaminant.sum()),
        }

    def submit(self, intents: Iterable[AnyIntent]) -> None:
        for intent in intents:
            if intent.tick < self.state.tick:
                self._reject(intent, ["scheduled in the past"])
            else:
                self.pending[intent.tick].append(intent)
                log.debug("queued %s from %s for tick %d", intent.kind, intent.agent, intent.tick)

    def _reject(self, intent: AnyIntent, violations: list[str]) -> None:
        reason = "; ".join(violations)
        self.rejected.append((intent, reason))
        log.warning("REJECTED %s from %s at tick %d: %s", intent.kind, intent.agent,
                    intent.tick, reason)  # fmt: skip
        message = f"{intent.agent}'s {intent.kind} was refused: {violations[0]}"
        self.events.emit(
            self.state.tick, "intent_rejected", message, agent=intent.agent, intent=intent.kind
        )

    def step(self) -> None:
        s, p = self.state, self.params

        for intent in self.pending.pop(s.tick, []):
            violations = self.validator.check(intent, s, self.accepted)
            if violations:
                self._reject(intent, violations)
                continue
            apply(intent, s, self.ledger, p)
            self.accepted.append(intent)
            log.info("APPLIED %s from %s at tick %d (%s)", intent.kind, intent.agent, s.tick,
                     describe(intent))  # fmt: skip
            self.events.emit(s.tick, "intent_applied", f"{intent.agent}: {describe(intent)}",
                             agent=intent.agent, intent=intent.kind)  # fmt: skip

        weather, s.temp_anomaly_c, s.raining = step_weather(
            s.tick, s.temp_anomaly_c, s.raining, p.weather, self.rng["weather"]
        )
        cover = canopy_cover(s.plant.c, p.plants)
        s.water, solutes = self.hydrology.step(
            s.water, s.layer_capacity, cover,
            Solutes(s.mineral_n, s.mineral_p, s.contaminant), weather, self.ledger,
        )  # fmt: skip
        s.mineral_n, s.mineral_p, s.contaminant = (
            solutes.mineral_n,
            solutes.mineral_p,
            solutes.contaminant,
        )

        ctx = TickContext(weather, s.moisture(), toxicity(s.contaminant, p.contamination),
                          self.ledger)  # fmt: skip
        gpp = photosynthesis(s, p.plants, ctx)
        step_decomposers(s, p.decomposers, ctx)
        degrade(s, p.contamination, ctx)
        delivered, paid = step_mycorrhiza(s, gpp, p.mycorrhiza, p.plants, ctx)
        root_uptake(s, p.plants, ctx)
        turnover(s, p.plants, self.rng["plants"], ctx)
        step_insects(s, p.insects, ctx)
        disperse_pool(s.plant, p.plants.dispersal_per_day * DT)
        for name, guild in (("bacteria", p.decomposers.bacteria),
                            ("saprotrophs", p.decomposers.saprotrophs),
                            ("mycorrhiza", p.mycorrhiza.guild)):  # fmt: skip
            if guild.dispersal_per_day:  # hyphal growth and cell spread
                disperse_pool(s.pool(name), guild.dispersal_per_day * DT)

        dep = p.nutrients.n_deposition_per_day * DT
        weathering = p.nutrients.p_weathering_per_day * DT
        s.mineral_n = s.mineral_n + dep
        s.mineral_p = s.mineral_p + weathering
        self.ledger.inflow("n", "deposition", dep * s.mineral_n.size)
        self.ledger.inflow("p", "weathering", weathering * s.mineral_p.size)

        s.atmosphere_c += ctx.atmosphere_delta
        s.tick += 1

        if self.check_balance:
            self.ledger.check(self.totals(), s.atmosphere_c, s.tick)
        self._record(weather, cover, gpp, delivered, paid)
        self.last_weather = weather
        self.detector.daily(self, self.events)
        if s.tick % 24 == 0 and log.isEnabledFor(logging.DEBUG):
            h, n = self.history, s.plant.c.size
            log.debug(
                "%s tick=%d T=%.1fC rain=%.1fmm moist=%.2f | plant=%.0f bact=%.1f sapro=%.1f "
                "myco=%.1f insect=%.2f som=%.0f | minN=%.3f trade=%.2fgC contam=%.0fg",
                date_label(s.tick),
                s.tick,
                sum(h["temp_c"][-24:]) / 24,
                sum(h["rain_mm"][-24:]),
                s.moisture().mean(),
                s.plant.c.mean(),
                s.bacteria.c.mean(),
                s.saprotrophs.c.mean(),
                s.mycorrhiza.c.mean(),
                s.insects.c.mean(),
                s.som.c.mean(),
                s.mineral_n.mean(),
                sum(h["trade_c"][-24:]) / n,
                s.contaminant.sum(),
            )

    def run_until(self, tick: int) -> None:
        while self.state.tick < tick:
            self.step()

    def state_hash(self) -> str:
        """World state hash plus every RNG stream's position.

        The world hash alone can miss divergence: a stream that has drifted but
        whose draws haven't changed any state yet (e.g. rare disturbances).
        """
        h = hashlib.sha256(self.state.state_hash().encode())
        for name in sorted(self.rng):
            h.update(json.dumps(self.rng[name].bit_generator.state, sort_keys=True).encode())
        return h.hexdigest()

    def run(self, ticks: int, checkpoint_every: int = 0) -> dict[int, str]:
        """Advance `ticks` ticks. Returns {tick: state hash} at each checkpoint and the end."""
        checkpoints: dict[int, str] = {}
        for _ in range(ticks):
            self.step()
            if checkpoint_every and self.state.tick % checkpoint_every == 0:
                checkpoints[self.state.tick] = self.state_hash()
        checkpoints[self.state.tick] = self.state_hash()
        return checkpoints

    def _record(self, weather, cover, gpp, delivered, paid) -> None:
        s, h = self.state, self.history
        h["temp_c"].append(weather.temp_c)
        h["shortwave"].append(weather.shortwave)
        h["rain_mm"].append(weather.rain_mm)
        h["atmosphere_c"].append(s.atmosphere_c)
        h["soil_water"].append(s.total_water())
        h["canopy_cover"].append(float(cover.mean()))
        h["mineral_n"].append(float(s.mineral_n.sum()))
        h["mineral_p"].append(float(s.mineral_p.sum()))
        h["contaminant"].append(float(s.contaminant.sum()))
        h["gpp"].append(float(np.sum(gpp)))
        h["trade_c"].append(float(paid.sum()))
        h["trade_n"].append(float(delivered["n"].sum()))
        h["trade_p"].append(float(delivered["p"].sum()))
        for pool in POOLS:
            h[f"{pool}_c"].append(s.pool(pool).total("c"))


def describe(intent: AnyIntent) -> str:
    """One-line human summary of an intent."""
    k = intent.kind
    if k == "shuttle_nutrients":
        return (f"shuttle {intent.amount_g:.2f} g {intent.element.upper()} "
                f"{intent.from_patch}->{intent.to_patch}")  # fmt: skip
    if k == "relocate_hyphae":
        return f"relocate {intent.fraction:.0%} of hyphae {intent.from_patch}->{intent.to_patch}"
    if k == "set_trade_bias":
        return f"set trade bias {intent.bias:.2f} in {intent.patch}"
    if k == "spill":
        return f"spill {intent.mass_g:.0f} g at ({intent.x:.0f},{intent.y:.0f})"
    if hasattr(intent, "x0"):
        box = f"x{intent.x0}-{intent.x1} y{intent.y0}-{intent.y1}"
        what = "plants" if k == "disturb" else "soil"
        return f"{k} {intent.fraction:.0%} of {what} in {box}"
    at = f"at ({intent.x:.0f},{intent.y:.0f}) r{intent.radius:g}"
    if k == "inoculate":
        return f"inoculate {intent.mass_c_g:.0f} g C of {intent.guild} {at}"
    if hasattr(intent, "water_mm"):
        return f"{k} {intent.water_mm:.0f} mm {at}"
    return f"{k} {intent.mass_c_g:.0f} g C {at}"
