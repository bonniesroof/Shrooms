"""The tick loop.

One tick = one sim-hour. Order within a tick is fixed:
    1. apply accepted intents scheduled for this tick
    2. weather
    3. soil water (uses pre-tick canopy)
    4. plant and soil carbon
    5. mass-balance check
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from sim import SIM_VERSION
from sim.intents import Disturb, InvalidIntent, apply, validate
from sim.ledger import Ledger
from sim.params import SimParams
from sim.plants import canopy_cover, step_carbon
from sim.rng import make_streams
from sim.water import step_water
from sim.weather import Weather, step_weather
from sim.world import WorldState, init_world

HISTORY_FIELDS = (
    "temp_c", "shortwave", "rain_mm",
    "plant_c", "soil_c", "atmosphere_c", "soil_water", "canopy_cover",
)  # fmt: skip


@dataclass
class Simulation:
    seed: int
    params: SimParams = field(default_factory=SimParams)
    check_balance: bool = True

    def __post_init__(self) -> None:
        self.rng = make_streams(self.seed)
        self.state: WorldState = init_world(self.params, self.rng["init"])
        self.ledger = Ledger(
            carbon_t0=self.state.total_carbon(),
            land_c_t0=self.state.land_carbon(),
            water_t0=self.state.total_water(),
        )
        self.pending: dict[int, list[Disturb]] = defaultdict(list)
        self.accepted: list[Disturb] = []
        self.rejected: list[tuple[Disturb, str]] = []
        self.history: dict[str, list[float]] = {k: [] for k in HISTORY_FIELDS}
        self.last_weather: Weather | None = None
        self.version = SIM_VERSION

    def submit(self, intents: Iterable[Disturb]) -> None:
        for intent in intents:
            if intent.tick < self.state.tick:
                self.rejected.append((intent, "scheduled in the past"))
            else:
                self.pending[intent.tick].append(intent)

    def step(self) -> None:
        s, p = self.state, self.params

        for intent in self.pending.pop(s.tick, []):
            try:
                validate(intent, s)
            except InvalidIntent as err:
                self.rejected.append((intent, str(err)))
                continue
            apply(intent, s, self.ledger)
            self.accepted.append(intent)

        weather, s.temp_anomaly_c, s.raining = step_weather(
            s.tick, s.temp_anomaly_c, s.raining, p.weather, self.rng["weather"]
        )
        cover = canopy_cover(s.plant_c, p.plants)
        s.soil_water = step_water(
            s.soil_water, s.water_capacity, cover, weather, p.water, self.ledger
        )
        s.plant_c, s.soil_c, s.atmosphere_c = step_carbon(
            s.plant_c, s.soil_c, s.atmosphere_c, s.soil_water, s.water_capacity,
            weather, p.plants, p.soil, self.rng["plants"], self.ledger,
        )  # fmt: skip
        s.tick += 1

        if self.check_balance:
            self.ledger.check(s.total_carbon(), s.land_carbon(), s.total_water(), s.tick)
        self._record(weather, cover)
        self.last_weather = weather

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

    def _record(self, weather: Weather, cover: np.ndarray) -> None:
        s, h = self.state, self.history
        h["temp_c"].append(weather.temp_c)
        h["shortwave"].append(weather.shortwave)
        h["rain_mm"].append(weather.rain_mm)
        h["plant_c"].append(float(s.plant_c.sum()))
        h["soil_c"].append(float(s.soil_c.sum()))
        h["atmosphere_c"].append(s.atmosphere_c)
        h["soil_water"].append(s.total_water())
        h["canopy_cover"].append(float(cover.mean()))
