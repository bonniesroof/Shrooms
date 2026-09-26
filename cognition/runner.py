"""Drive a sim with agents on slow clocks, and save everything a run produces.

The sim advances in segments between clock ticks: mycelium decisions (weekly by
default), narrator entries (monthly) and replay checkpoints. Agents decide at
tick T; their intents apply at T + latency. In this headless runner the sim
pauses while an agent thinks; in Phase 3's server it will keep running and
the latency will be real wall-clock time.

Artifacts per run (in out_dir, named by stem):
    <stem>.json           replay record: seed, version, params, accepted intents
    <stem>_journal.md     the Narrator's field journal
    <stem>_trace.jsonl    one line per mycelium decision (observation, reasoning, verdicts)
    <stem>_agents.sqlite  LangGraph checkpoints for both agents
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from cognition.keystone import MyceliumAgent
from cognition.llm import Router
from cognition.narrator import Narrator
from sim import SIM_VERSION
from sim.engine import Simulation
from sim.params import SimParams
from sim.replay import ReplayRecord

log = logging.getLogger("shrooms.runner")


@dataclass
class AgentConfig:
    mycelium_every: int = 168  # ticks (one sim-week)
    narrator_every: int = 720  # ticks (30 sim-days, so entries cover whole days)
    checkpoint_every: int = 730
    latency_ticks: int = 1
    enable_mycelium: bool = True
    enable_narrator: bool = True


@dataclass
class RunArtifacts:
    sim: Simulation
    record: ReplayRecord
    mycelium: MyceliumAgent | None
    narrator: Narrator | None
    paths: dict[str, Path] = field(default_factory=dict)


def _next(tick: int, every: int) -> int:
    return (tick // every + 1) * every


def run_with_agents(
    seed: int,
    ticks: int,
    router: Router,
    out_dir: Path,
    stem: str,
    params: SimParams | None = None,
    config: AgentConfig | None = None,
) -> RunArtifacts:
    cfg = config or AgentConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "record": out_dir / f"{stem}.json",
        "journal": out_dir / f"{stem}_journal.md",
        "trace": out_dir / f"{stem}_trace.jsonl",
        "checkpoints": out_dir / f"{stem}_agents.sqlite",
    }
    for key in ("journal", "trace", "checkpoints"):
        paths[key].unlink(missing_ok=True)

    conn = sqlite3.connect(paths["checkpoints"], check_same_thread=False)
    saver = SqliteSaver(conn)
    sim = Simulation(seed, params or SimParams())
    myc = (
        MyceliumAgent(router, saver, cfg.latency_ticks, thread_id=f"{stem}:mycelium")
        if cfg.enable_mycelium
        else None
    )
    nar = (
        Narrator(router, paths["journal"], saver, thread_id=f"{stem}:narrator")
        if cfg.enable_narrator
        else None
    )
    log.info(
        "run %s: seed %d, %d ticks, sim %s, mycelium every %d, narrator every %d",
        stem,
        seed,
        ticks,
        SIM_VERSION,
        cfg.mycelium_every,
        cfg.narrator_every,
    )

    checkpoints: dict[int, str] = {}
    with paths["trace"].open("w") as trace:
        while sim.state.tick < ticks:
            t = sim.state.tick
            stop = min(
                ticks,
                _next(t, cfg.checkpoint_every),
                _next(t, cfg.mycelium_every) if myc else ticks,
                _next(t, cfg.narrator_every) if nar else ticks,
            )
            sim.run_until(stop)
            t = sim.state.tick
            if t % cfg.checkpoint_every == 0:
                checkpoints[t] = sim.state_hash()
            if myc and t % cfg.mycelium_every == 0 and t < ticks:
                myc.decide(sim)
                trace.write(json.dumps(myc.traces[-1], default=str) + "\n")
                trace.flush()
            if nar and (t % cfg.narrator_every == 0 or t == ticks):
                nar.narrate(sim)
    checkpoints[sim.state.tick] = sim.state_hash()
    conn.close()

    record = ReplayRecord(
        sim_version=SIM_VERSION,
        seed=seed,
        ticks=ticks,
        params=sim.params,
        intents=sim.accepted,
        checkpoints=checkpoints,
    )
    record.save(paths["record"])
    log.info(
        "run %s done: %d intents accepted, %d rejected at apply time",
        stem,
        len(sim.accepted),
        len(sim.rejected),
    )
    return RunArtifacts(sim, record, myc, nar, paths)
