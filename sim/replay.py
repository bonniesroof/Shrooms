"""Replay records.

A run is fully described by its seed, sim version, parameters and the intents
accepted (with the tick each was applied). Checkpoint hashes let a replay
report the first tick range where it diverged.
"""

import json
import warnings
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sim import SIM_VERSION
from sim.engine import Simulation
from sim.intents import Intent
from sim.params import SimParams


class ReplayMismatch(AssertionError):
    pass


class ReplayRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sim_version: str
    seed: int
    ticks: int
    params: SimParams
    intents: list[Intent]
    checkpoints: dict[int, str]  # tick -> state hash

    @property
    def final_hash(self) -> str:
        return self.checkpoints[max(self.checkpoints)]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ReplayRecord":
        return cls.model_validate(json.loads(Path(path).read_text()))


def record_run(
    seed: int,
    ticks: int,
    params: SimParams | None = None,
    intents: list | None = None,
    checkpoint_every: int = 730,
) -> tuple[Simulation, ReplayRecord]:
    sim = Simulation(seed, params or SimParams())
    sim.submit(intents or [])
    checkpoints = sim.run(ticks, checkpoint_every)
    record = ReplayRecord(
        sim_version=SIM_VERSION,
        seed=seed,
        ticks=ticks,
        params=sim.params,
        intents=sim.accepted,
        checkpoints=checkpoints,
    )
    return sim, record


def replay(record: ReplayRecord) -> Simulation:
    """Re-run a record and raise ReplayMismatch at the first diverging checkpoint."""
    if record.sim_version != SIM_VERSION:
        warnings.warn(
            f"record is sim {record.sim_version}, running sim {SIM_VERSION}",
            stacklevel=2,
        )
    sim = Simulation(record.seed, record.params)
    sim.submit(record.intents)
    previous = 0
    for tick in sorted(record.checkpoints):
        sim.run(tick - sim.state.tick)
        got = sim.state.state_hash()
        if got != record.checkpoints[tick]:
            raise ReplayMismatch(f"diverged between tick {previous} and {tick}")
        previous = tick
    return sim
