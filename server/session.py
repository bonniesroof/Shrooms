"""A live game session: the sim, its agents, the director, the scenario, the player.

The sim never waits on an LLM. When an agent is due to decide, it gets a snapshot
of the sim and thinks in a worker thread while the live sim keeps running. Its
intents are applied when the decision lands, at the next tick: the latency is
real wall-clock time. (With the scripted policy, or `sync_agents=True`, agents
decide inline, which keeps tests deterministic.)

Everything that changes the world (player tools, agent actions, director
events, scenario setup) is a typed intent, validated by the engine and saved
in the replay record, so any session replays exactly without calling a model.
"""

import base64
import copy
import logging
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cognition import scripted
from cognition.keystone import MyceliumAgent
from cognition.llm import Router, routes_from_spec
from cognition.narrator import Narrator
from server.director import Director
from server.forecasts import LiveForecasts
from server.scenarios import SCENARIOS, Brownfield
from sim import SIM_VERSION, TICKS_PER_DAY
from sim.engine import Simulation, describe
from sim.events import date_label
from sim.intents import (
    Amend,
    Disturb,
    Excavate,
    Inoculate,
    Irrigate,
    Seed,
    Spill,
)
from sim.params import SimParams
from sim.replay import ReplayRecord

log = logging.getLogger("shrooms.session")

MYCELIUM_EVERY = 168  # one sim-week
NARRATOR_EVERY = 720  # 30 sim-days
SPEEDS = (0, 6, 24, 96, 336)  # ticks per real second: paused, 1/4 day, 1 day, 4 days, 2 weeks


@dataclass(frozen=True)
class Tool:
    label: str
    cost: float
    make: object  # (x, y, tick) -> intent
    research_only: bool = False
    hint: str = ""


def _box(x: float, y: float, half: int, n: int = 64) -> dict:
    x, y = int(x), int(y)
    return {"x0": max(0, x - half), "y0": max(0, y - half),
            "x1": min(n, x + half), "y1": min(n, y + half)}  # fmt: skip


TOOLS: dict[str, Tool] = {
    "inoculate_bacteria": Tool(
        "Bacteria",
        15,
        lambda x, y, t: Inoculate(tick=t, x=x, y=y, radius=4, guild="bacteria", mass_c_g=300.0),
        hint="Cultured degraders: they eat the contaminant",
    ),  # fmt: skip
    "inoculate_fungi": Tool(
        "Fungi",
        15,
        lambda x, y, t: Inoculate(tick=t, x=x, y=y, radius=4, guild="saprotrophs", mass_c_g=300.0),
        hint="White-rot fungi: tough, and degrade it too",
    ),  # fmt: skip
    "inoculate_mycorrhiza": Tool(
        "Mycorrhiza",
        15,
        lambda x, y, t: Inoculate(tick=t, x=x, y=y, radius=4, guild="mycorrhiza", mass_c_g=150.0),
        hint="Seed the network where it can't reach",
    ),  # fmt: skip
    "compost": Tool(
        "Compost",
        10,
        lambda x, y, t: Amend(tick=t, x=x, y=y, radius=5, mass_c_g=4000.0),
        hint="Feeds decomposers; nitrogen for plants",
    ),  # fmt: skip
    "seed": Tool(
        "Seed",
        8,
        lambda x, y, t: Seed(tick=t, x=x, y=y, radius=5, mass_c_g=800.0),
        hint="Sow a meadow mix",
    ),  # fmt: skip
    "irrigate": Tool(
        "Water",
        3,
        lambda x, y, t: Irrigate(tick=t, x=x, y=y, radius=5, water_mm=30.0),
        hint="30 mm of water",
    ),  # fmt: skip
    "excavate": Tool(
        "Excavate",
        60,
        lambda x, y, t: Excavate(tick=t, fraction=0.6, **_box(x, y, 4)),
        hint="Dig out and haul away 60% of an 8x8 m block",
    ),  # fmt: skip
    "mow": Tool(
        "Mow",
        1,
        lambda x, y, t: Disturb(tick=t, fraction=0.8, **_box(x, y, 3)),
        hint="Cut plants to litter",
    ),  # fmt: skip
    "spill": Tool(
        "Spill",
        0,
        lambda x, y, t: Spill(tick=t, x=x, y=y, radius=3, mass_g=500.0),
        research_only=True,
        hint="Research only: contaminate a spot",
    ),  # fmt: skip
}

# Fixed scales so colours mean the same thing all game. Values are clipped.
FIELD_SCALES = {
    "plant": 1000.0, "moisture": 1.0, "contaminant": 60.0, "mycorrhiza": 30.0,
    "bacteria": 30.0, "saprotrophs": 40.0, "insects": 1.5, "litter": 800.0,
    "som": 2500.0, "mineral_n": 0.05, "elevation": 4.0,
}  # fmt: skip


def snapshot(sim: Simulation) -> Simulation:
    """A copy an agent can read while the live sim keeps stepping."""
    snap = copy.copy(sim)
    snap.state = copy.deepcopy(sim.state)
    snap.pending = defaultdict(list)
    snap.accepted = list(sim.accepted)
    snap.rejected = list(sim.rejected)
    return snap


def _field(state, name: str) -> np.ndarray:
    if name == "moisture":
        return state.moisture()
    if name in ("contaminant", "mineral_n", "elevation"):
        return getattr(state, name)
    return state.pool(name).c


def encode(values: np.ndarray, scale: float) -> str:
    q = np.clip(values / scale, 0.0, 1.0) * 255.0
    return base64.b64encode(np.round(q).astype(np.uint8).tobytes()).decode()


class GameSession:
    def __init__(
        self,
        seed: int = 42,
        scenario: str | None = "brownfield",
        mode: str = "game",
        router: Router | None = None,
        agents: bool = True,
        sync_agents: bool | None = None,
        out_dir: Path = Path("runs/sessions"),
    ):
        if mode not in ("game", "research"):
            raise ValueError(f"mode must be game or research, not {mode!r}")
        self.seed, self.mode = seed, mode
        self.scenario: Brownfield | None = SCENARIOS[scenario]() if scenario else None
        params = self.scenario.params() if self.scenario else SimParams()
        self.sim = Simulation(seed, params)
        if self.scenario:
            self.sim.submit(self.scenario.setup_intents())
        self.router = router or Router(routes_from_spec("scripted"), scripted=scripted.POLICIES)
        scripted_only = all(
            route.backend == "scripted" for chain in self.router.routes.values() for route in chain
        )
        self.sync_agents = scripted_only if sync_agents is None else sync_agents
        out_dir.mkdir(parents=True, exist_ok=True)
        self.stem = f"{scenario or 'meadow'}_seed{seed}"
        self.mycelium = (
            MyceliumAgent(self.router, thread_id=f"{self.stem}:mycelium") if agents else None
        )
        self.narrator = (
            Narrator(self.router, out_dir / f"{self.stem}_journal.md",
                     thread_id=f"{self.stem}:narrator") if agents else None
        )  # fmt: skip
        self.director = (
            Director(seed, self.scenario.site, self.scenario.deadline_days)
            if self.scenario else None
        )  # fmt: skip
        self.budget = self.scenario.start_budget if self.scenario else 0.0
        self.research_used = mode == "research"
        self.status = self.scenario.evaluate(self.sim) if self.scenario else None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent")
        self.inflight: tuple[str, Future] | None = None
        self.lock = threading.RLock()
        self.journal: list[dict] = []
        self.feed: list[dict] = []  # player-facing messages
        self.decision_id = 0
        self.forecasts = LiveForecasts()

    # --- time ---------------------------------------------------------------

    def advance(self, ticks: int) -> None:
        """Step the sim up to `ticks` ticks, running the clocks due along the way."""
        with self.lock:
            for _ in range(ticks):
                if self.status and self.status["status"] != "playing":
                    return
                self._collect_decision()
                self.sim.step()
                t = self.sim.state.tick
                if t % TICKS_PER_DAY == 0:
                    self._daily()
                if self.mycelium and t % MYCELIUM_EVERY == 0:
                    self._start("mycelium")
                if self.narrator and t % NARRATOR_EVERY == 0:
                    self._start("narrator")

    def _daily(self) -> None:
        self.forecasts.daily(self.sim)
        if not self.scenario:
            return
        day = self.sim.state.tick // TICKS_PER_DAY
        if self.mode == "game" and day % self.scenario.grant_every_days == 0:
            self.budget += self.scenario.grant
        self.status = self.scenario.evaluate(self.sim)
        if self.mode == "game" and self.director:
            for intent in self.director.daily(self.sim, self.status):
                self.sim.submit([intent])
                self._say("director", intent.rationale.removeprefix("director: "))
        if self.status["status"] != "playing":
            self._say("scenario", f"Scenario {self.status['status']} on day {day}.")

    # --- agents -------------------------------------------------------------

    def _start(self, who: str) -> None:
        if self.inflight is not None:  # one thinker at a time; skip if still busy
            log.info("%s decision skipped: another agent is still thinking", who)
            return
        agent = self.mycelium if who == "mycelium" else self.narrator
        call = agent.decide if who == "mycelium" else agent.narrate
        if self.sync_agents:
            self._land(who, call(self.sim) if who == "mycelium" else call(snapshot(self.sim)))
        else:
            self.inflight = (who, self.executor.submit(call, snapshot(self.sim)))

    def _collect_decision(self) -> None:
        if self.inflight and self.inflight[1].done():
            who, fut = self.inflight
            self.inflight = None
            self._land(who, fut.result())

    def _land(self, who: str, result) -> None:
        if who == "narrator":
            if self.narrator.entries:
                self.journal.append(self.narrator.entries[-1])
            return
        now = self.sim.state.tick
        late = [i.model_copy(update={"tick": max(i.tick, now)}) for i in result]
        if not self.sync_agents:  # the sync path already committed through the agent
            self.sim.submit(late)
        self.decision_id += 1

    def whisper(self, text: str) -> None:
        if self.mycelium:
            self.mycelium.whisper(text)
            self._say("you", f'whispered to the network: "{text.strip()[:300]}"')

    # --- the player ---------------------------------------------------------

    def use_tool(self, name: str, x: float, y: float) -> dict:
        with self.lock:
            tool = TOOLS.get(name)
            if tool is None:
                return {"ok": False, "error": f"unknown tool {name!r}"}
            if tool.research_only and self.mode != "research":
                return {"ok": False, "error": f"{tool.label} is only available in research mode"}
            if self.status and self.status["status"] != "playing":
                return {"ok": False, "error": f"the scenario is over ({self.status['status']})"}
            cost = tool.cost if (self.mode == "game" and self.scenario) else 0.0
            if cost > self.budget + 1e-9:
                return {
                    "ok": False,
                    "error": f"{tool.label} costs {cost:g}; budget is {self.budget:g}",
                }
            intent = tool.make(float(x), float(y), self.sim.state.tick)
            violations = self.sim.validator.check(intent, self.sim.state, self.sim.accepted)
            if violations:
                return {"ok": False, "error": violations[0]}
            self.budget -= cost
            self.sim.submit([intent])
            self._say("you", describe(intent) + (f" (-{cost:g})" if cost else ""))
            return {"ok": True, "cost": cost, "budget": self.budget}

    def set_mode(self, mode: str) -> None:
        if mode not in ("game", "research"):
            raise ValueError(mode)
        self.mode = mode
        if mode == "research":
            self.research_used = True
        self._say("session", f"switched to {mode} mode"
                  + (" (this run is now unranked)" if mode == "research" else ""))  # fmt: skip

    def _say(self, who: str, text: str) -> None:
        self.feed.append({"tick": self.sim.state.tick, "date": date_label(self.sim.state.tick),
                          "who": who, "text": text})  # fmt: skip
        self.feed = self.feed[-200:]

    # --- views --------------------------------------------------------------

    def frame(self, fields: tuple[str, ...] = tuple(FIELD_SCALES), since: int = 0) -> dict:
        with self.lock:
            s = self.sim.state
            h, w = s.shape
            events = [e.as_dict() for e in self.sim.events.since(since)
                      if not e.kind.startswith("intent_applied")][-30:]  # fmt: skip
            return {
                "type": "frame",
                "tick": s.tick,
                "date": date_label(s.tick),
                "mode": self.mode,
                "research_used": self.research_used,
                "grid": [h, w],
                "fields": {k: encode(_field(s, k), FIELD_SCALES[k]) for k in fields},
                "scales": {k: FIELD_SCALES[k] for k in fields},
                "stats": self.stats(),
                "scenario": self.scenario_view(),
                "budget": round(self.budget, 1),
                "flows": self.network_flows(),
                "events": events,
                "feed": self.feed[-40:],
                "journal": self.journal[-5:],
                "inspector": self.inspector(),
                "thinking": self.inflight[0] if self.inflight else None,
                "forecast": self.forecasts.view(),
            }

    def stats(self) -> dict:
        s = self.sim.state
        return {
            "plant_c": round(float(s.plant.c.mean()), 1),
            "fungal_c": round(float(s.mycorrhiza.c.mean()), 2),
            "bacteria_c": round(float(s.bacteria.c.mean()), 2),
            "insects_c": round(float(s.insects.c.mean()), 3),
            "contaminant_kg": round(float(s.contaminant.sum()) / 1000, 2),
            "moisture": round(float(s.moisture().mean()), 2),
            "temp_c": round(self.sim.last_weather.temp_c, 1) if self.sim.last_weather else None,
            "raining": bool(s.raining),
        }

    def scenario_view(self) -> dict | None:
        if not self.scenario:
            return None
        return {"name": self.scenario.name, "title": self.scenario.title,
                "site": self.scenario.site, **(self.status or {})}  # fmt: skip

    def network_flows(self, window: int = 2 * MYCELIUM_EVERY) -> list[dict]:
        t = self.sim.state.tick
        return [
            {"kind": i.kind, "from": getattr(i, "from_patch", None),
             "to": getattr(i, "to_patch", None) or getattr(i, "patch", None),
             "element": getattr(i, "element", None), "tick": i.tick}
            for i in self.sim.accepted[-60:]
            if i.agent == "mycelium" and t - i.tick <= window
        ]  # fmt: skip

    def inspector(self) -> dict | None:
        """The latest keystone decision: observation -> intent -> generated rationale."""
        if not self.mycelium or not self.mycelium.traces:
            return None
        tr = self.mycelium.traces[-1]
        obs = tr.get("observation") or {}
        return {
            "id": len(self.mycelium.traces),
            "date": date_label(tr["tick"]),
            "whisper": tr.get("whisper"),
            "observation": {
                "network": obs.get("network"),
                "market": obs.get("market"),
                "needy_partners": (obs.get("needy_partners") or [])[:3],
                "contaminated": (obs.get("contaminated") or [])[:3],
            },
            "deliberation": tr.get("deliberation"),
            "accepted": tr.get("accepted") or [],
            "rejected": tr.get("rejected") or [],
            "attempts": tr.get("attempt"),
            "routes": tr.get("routes"),
        }

    def record(self) -> ReplayRecord:
        with self.lock:
            return ReplayRecord(
                sim_version=SIM_VERSION, seed=self.seed, ticks=self.sim.state.tick,
                params=self.sim.params, intents=list(self.sim.accepted),
                checkpoints={self.sim.state.tick: self.sim.state_hash()},
            )  # fmt: skip

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
