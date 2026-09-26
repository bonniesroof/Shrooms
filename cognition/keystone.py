"""The mycelial-network keystone overlay, as a LangGraph graph.

    observe -> deliberate -> propose -> validate --(rejections, retries left)--> deliberate
                                               \\--(otherwise)--> commit

- observe: compact observation of the sim (cognition/observe.py)
- deliberate: free-text reasoning from the model
- propose: a JSON list of intents from the model
- validate: schema parsing, then the sim's rule validator run on a scratch copy of
  the state, applying accepted proposals in order so they can't jointly overdraw
- commit: accepted intents go to the sim for `tick + latency`, where the engine
  validates them again (authoritatively) before applying

Graph state is checkpointed to SQLite after every node, per run thread.
"""

import copy
import json
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import TypeAdapter, ValidationError

from cognition.llm import Router
from cognition.observe import observe_mycelium, patch_table
from cognition.prompts import DELIBERATE, FEEDBACK, MYCELIUM_SYSTEM, PROPOSE, as_json
from sim.engine import Simulation, describe
from sim.events import date_label
from sim.intents import Intent, apply
from sim.ledger import Ledger

log = logging.getLogger("shrooms.agent.mycelium")
INTENT = TypeAdapter(Intent)


class KeystoneState(TypedDict, total=False):
    tick: int
    observation: dict
    deliberation: str
    proposals: list[dict]
    accepted: list[dict]
    rejected: list[dict]
    new_rejections: int  # from the latest validate pass
    attempt: int
    routes: list[str]


class MyceliumAgent:
    name = "mycelium"

    def __init__(
        self,
        router: Router,
        checkpointer=None,
        latency_ticks: int = 1,
        max_attempts: int = 2,
        thread_id: str = "mycelium",
    ):
        self.router = router
        self.latency = latency_ticks
        self.max_attempts = max_attempts
        self.config = {"configurable": {"thread_id": thread_id}}
        self.sim: Simulation | None = None
        self.watch: list[dict] = []  # committed actions whose outcome we report next time
        self.traces: list[dict] = []
        self.pending_whisper: str | None = None  # human-in-the-loop suggestion for next decision
        g = StateGraph(KeystoneState)
        for node in ("observe", "deliberate", "propose", "validate", "commit"):
            g.add_node(node, getattr(self, f"_{node}"))
        g.add_edge(START, "observe")
        g.add_edge("observe", "deliberate")
        g.add_edge("deliberate", "propose")
        g.add_edge("propose", "validate")
        g.add_conditional_edges("validate", self._route_after_validate, ["deliberate", "commit"])
        g.add_edge("commit", END)
        self.graph = g.compile(checkpointer=checkpointer)

    # --- public -------------------------------------------------------------

    def whisper(self, text: str) -> None:
        """Queue a suggestion from the player; the next decision sees it once."""
        self.pending_whisper = text.strip()[:300] or None

    def decide(self, sim: Simulation) -> list:
        self.sim = sim
        whisper, self.pending_whisper = self.pending_whisper, None
        self._whisper = whisper
        log.info("--- decision at %s (tick %d) ---", date_label(sim.state.tick), sim.state.tick)
        final = self.graph.invoke(
            {
                "tick": sim.state.tick,
                "attempt": 0,
                "accepted": [],
                "rejected": [],
                "new_rejections": 0,
                "routes": [],
            },
            self.config,
        )
        committed = [INTENT.validate_python(d) for d in final["accepted"]]
        self.traces.append(
            {"whisper": whisper}
            | {
                k: final.get(k)
                for k in (
                    "tick",
                    "observation",
                    "deliberation",
                    "accepted",
                    "rejected",
                    "attempt",
                    "routes",
                )
            }
        )
        log.info(
            "decision at tick %d: %d committed, %d rejected, %d attempt(s)",
            sim.state.tick,
            len(committed),
            len(final["rejected"]),
            final["attempt"],
        )
        return committed

    # --- nodes --------------------------------------------------------------

    def _observe(self, state: KeystoneState) -> dict:
        obs = observe_mycelium(self.sim, self._outcomes(), self.latency)
        if getattr(self, "_whisper", None):
            obs["whisper"] = self._whisper
        log.info(
            "observe: %d/%d patches on network; neediest %s; richest store %s",
            obs["network"]["patches_on_network"],
            obs["network"]["patches_total"],
            obs["needy_partners"][0]["patch"] if obs["needy_partners"] else "-",
            obs["nutrient_stores"][0]["patch"] if obs["nutrient_stores"] else "-",
        )
        log.debug("observation: %s", json.dumps(obs, separators=(",", ":")))
        return {"observation": obs}

    def _deliberate(self, state: KeystoneState) -> dict:
        messages = [
            {"role": "system", "content": MYCELIUM_SYSTEM},
            {
                "role": "user",
                "content": DELIBERATE.format(
                    observation=as_json(state["observation"]), feedback=self._feedback(state)
                ),
            },
        ]
        res = self.router.complete(self.name, messages, context=self._ctx(state, "deliberate"))
        text = res.text.strip()
        log.info("deliberate [%s]: %s", res.route.label, text.replace("\n", " ")[:300])
        return {"deliberation": text, "routes": [*state.get("routes", []), res.route.label]}

    def _propose(self, state: KeystoneState) -> dict:
        messages = [
            {"role": "system", "content": MYCELIUM_SYSTEM},
            {
                "role": "user",
                "content": PROPOSE.format(
                    observation=as_json(state["observation"]),
                    deliberation=state["deliberation"],
                    feedback=self._feedback(state),
                ),
            },
        ]
        res = self.router.complete(
            self.name, messages, context=self._ctx(state, "propose"), want_json=True
        )
        parsed = res.parsed
        raw = parsed.get("intents", []) if isinstance(parsed, dict) else parsed
        proposals = [p for p in (raw or []) if isinstance(p, dict)]
        log.info("propose [%s]: %d proposal(s)", res.route.label, len(proposals))
        for p in proposals:
            log.debug("  proposal: %s", p)
        return {
            "proposals": proposals,
            "attempt": state["attempt"] + 1,
            "routes": [*state.get("routes", []), res.route.label],
        }

    def _validate(self, state: KeystoneState) -> dict:
        sim = self.sim
        act_tick = sim.state.tick + self.latency
        scratch = copy.deepcopy(sim.state)
        scratch_ledger = Ledger({k: 0.0 for k in ("c", "n", "p", "water", "contaminant")}, 0.0)
        accepted = list(state["accepted"])
        history = list(sim.accepted) + [INTENT.validate_python(d) for d in accepted]
        for d in history[len(sim.accepted) :]:  # replay this decision's accepted on the scratch
            apply(d, scratch, scratch_ledger, sim.params)
        rejected = []
        for raw in state["proposals"]:
            candidate = {**raw, "tick": act_tick, "agent": self.name}
            candidate.setdefault("rationale", "")
            try:
                intent = INTENT.validate_python(candidate)
            except ValidationError as err:
                problems = [
                    f"schema: {e['loc'][-1] if e['loc'] else ''} {e['msg']}" for e in err.errors()
                ]
                rejected.append({"proposal": raw, "violations": problems})
                log.warning("validate: REJECT %s -> %s", raw, "; ".join(problems))
                continue
            violations = sim.validator.check(intent, scratch, history)
            if violations:
                rejected.append({"proposal": raw, "violations": violations})
                log.warning("validate: REJECT %s -> %s", describe(intent), "; ".join(violations))
                continue
            apply(intent, scratch, scratch_ledger, sim.params)
            history.append(intent)
            accepted.append(intent.model_dump())
            log.info("validate: accept %s", describe(intent))
        return {
            "accepted": accepted,
            "rejected": [*state["rejected"], *rejected],
            "new_rejections": len(rejected),
        }

    def _route_after_validate(self, state: KeystoneState) -> str:
        new = state.get("new_rejections", 0)
        retry = new > 0 and state["attempt"] < self.max_attempts
        if retry:
            log.info(
                "validate: %d rejection(s); re-deliberating (attempt %d of %d)",
                new,
                state["attempt"] + 1,
                self.max_attempts,
            )
        return "deliberate" if retry else "commit"

    def _commit(self, state: KeystoneState) -> dict:
        intents = [INTENT.validate_python(d) for d in state["accepted"]]
        self.sim.submit(intents)
        table = patch_table(self.sim)
        self.watch = [
            {
                "action": describe(i),
                "patch": t,
                "plant_cn_then": table[t]["plant_cn"],
                "fungal_c_then": table[t]["fungal_c"],
            }
            for i in intents
            if (t := getattr(i, "to_patch", None) or getattr(i, "patch", None))
        ]
        log.info(
            "commit: %d intent(s) for tick %d", len(intents), self.sim.state.tick + self.latency
        )
        return {}

    # --- helpers ------------------------------------------------------------

    def _ctx(self, state: KeystoneState, step: str) -> dict:
        blocked = [
            f"{r['proposal'].get('kind')}->"
            f"{r['proposal'].get('to_patch') or r['proposal'].get('patch')}"
            for r in state.get("rejected", [])
        ]
        blocked += [
            f"{a['kind']}->{a.get('to_patch') or a.get('patch')}" for a in state.get("accepted", [])
        ]
        return {"step": step, "observation": state["observation"], "blocked": blocked}

    @staticmethod
    def _feedback(state: KeystoneState) -> str:
        if not state.get("rejected"):
            return ""
        lines = [f"- {r['proposal']}: {'; '.join(r['violations'])}" for r in state["rejected"]]
        return FEEDBACK.format(rejected="\n".join(lines))

    def _outcomes(self) -> list[dict]:
        if not self.watch:
            return []
        table = patch_table(self.sim)
        return [
            {
                **w,
                "plant_cn_now": table[w["patch"]]["plant_cn"],
                "fungal_c_now": table[w["patch"]]["fungal_c"],
            }
            for w in self.watch
        ]
