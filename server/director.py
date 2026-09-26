"""Scenario Director (game mode only): pacing through events.

It watches how the player is doing against the clock and nudges the story:

    ahead of schedule  -> a setback (a buried drum leaks; a pest swarm arrives)
    behind schedule    -> relief in a dry spell (a summer cloudburst)
    on schedule        -> occasional ambient weather

Every event is a typed intent from agent "director", validated like any other
and saved in the replay record, so a directed game replays exactly without
the director. There is a cooldown between events, and the director never acts
in research mode.
"""

from dataclasses import dataclass

from sim import TICKS_PER_DAY
from sim.engine import Simulation
from sim.intents import AnyIntent, Downpour, PestOutbreak, Spill
from sim.rng import stream


@dataclass
class Director:
    seed: int
    site: tuple[int, int, int, int]
    deadline_days: int
    cooldown_days: int = 45
    base_chance: float = 0.03  # per day, when off cooldown

    def __post_init__(self) -> None:
        self.rng = stream(self.seed, "director")
        self.last_day = -10_000
        self.log: list[dict] = []

    def daily(self, sim: Simulation, status: dict) -> list[AnyIntent]:
        day = sim.state.tick // TICKS_PER_DAY
        roll, px, py = self.rng.random(), self.rng.random(), self.rng.random()  # fixed draws
        if day - self.last_day < self.cooldown_days or day < 30:
            return []
        schedule = min(1.0, day / (0.8 * self.deadline_days))
        lead = status["progress"] - schedule
        season = (day % 365) / 365
        summer = 0.35 < season < 0.7
        dry = float(sim.state.moisture().mean()) < 0.45

        x0, y0, x1, y1 = self.site
        x, y = x0 + px * (x1 - x0), y0 + py * (y1 - y0)
        t, d = sim.state.tick + 1, {"agent": "director"}
        intent: AnyIntent | None = None
        if lead > 0.15 and roll < 3 * self.base_chance:
            why = "director: you're ahead - "
            if roll < 1.5 * self.base_chance:
                intent = Spill(tick=t, x=x, y=y, radius=2.5, mass_g=600.0,
                               rationale=why + "a buried drum gives way", **d)  # fmt: skip
            elif summer:
                intent = PestOutbreak(tick=t, x=x, y=y, radius=6, mass_c_g=40.0,
                                      rationale=why + "a pest swarm arrives", **d)  # fmt: skip
        elif lead < -0.2 and dry and roll < 4 * self.base_chance:
            intent = Downpour(tick=t, x=32, y=32, radius=14, water_mm=45.0,
                              rationale="director: you're behind - rain breaks the drought",
                              **d)  # fmt: skip
        elif summer and roll < self.base_chance:
            intent = Downpour(tick=t, x=x, y=y, radius=10, water_mm=35.0,
                              rationale="director: a summer thunderstorm", **d)  # fmt: skip
        if intent is None:
            return []
        self.last_day = day
        self.log.append({"day": day, "kind": intent.kind, "lead": round(lead, 2)})
        return [intent]
