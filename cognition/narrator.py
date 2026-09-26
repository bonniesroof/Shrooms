"""The Narrator, as a LangGraph graph: reads the event log, writes the field journal.

    gather -> draft -> check --(problems, retries left)--> draft
                           \\--(otherwise)--> write

`check` is a rule-based fact check: the entry must be non-empty and short, and
every number in it must appear in the facts it was given (within rounding). A
draft that still fails after the retry is replaced by the scripted template and
the entry is marked as such, so an unreliable model can't put fiction in the
journal. The Narrator never proposes intents; it only reads.
"""

import json
import logging
import re
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from cognition import scripted
from cognition.llm import Router
from cognition.observe import narrator_facts
from cognition.prompts import NARRATE, NARRATE_FEEDBACK, NARRATOR_SYSTEM, as_json
from sim.engine import Simulation

log = logging.getLogger("shrooms.agent.narrator")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
MAX_CHARS = 1200


class NarratorState(TypedDict, total=False):
    since_tick: int
    facts: dict
    draft: str
    problems: list[str]
    attempt: int
    route: str
    grounded: bool


def fact_numbers(facts) -> list[float]:
    return [float(x) for x in NUMBER.findall(as_json(facts))]


def check_entry(text: str, facts: dict) -> list[str]:
    problems = []
    if not text.strip():
        return ["entry is empty"]
    if len(text) > MAX_CHARS:
        problems.append(f"entry is {len(text)} characters (max {MAX_CHARS})")
    known = fact_numbers(facts)
    for raw in NUMBER.findall(text):
        x = float(raw)
        if not any(abs(x - k) <= max(0.051 * abs(k), 0.5) for k in known):
            problems.append(f"number {raw} is not in the facts")
    return problems


class Narrator:
    name = "narrator"

    def __init__(
        self,
        router: Router,
        journal_path: Path,
        checkpointer=None,
        max_attempts: int = 2,
        thread_id: str = "narrator",
    ):
        self.router = router
        self.path = Path(journal_path)
        self.max_attempts = max_attempts
        self.config = {"configurable": {"thread_id": thread_id}}
        self.sim: Simulation | None = None
        self.last_tick = 0
        self.entries: list[dict] = []
        g = StateGraph(NarratorState)
        for node in ("gather", "draft", "check", "write"):
            g.add_node(node, getattr(self, f"_{node}"))
        g.add_edge(START, "gather")
        g.add_edge("gather", "draft")
        g.add_edge("draft", "check")
        g.add_conditional_edges("check", self._after_check, ["draft", "write"])
        g.add_edge("write", END)
        self.graph = g.compile(checkpointer=checkpointer)

    def narrate(self, sim: Simulation) -> dict:
        self.sim = sim
        final = self.graph.invoke(
            {"since_tick": self.last_tick, "attempt": 0, "problems": []}, self.config
        )
        self.last_tick = sim.state.tick
        return final

    def _gather(self, state: NarratorState) -> dict:
        facts = narrator_facts(self.sim, state["since_tick"])
        log.info("gather: %s, %d event(s)", facts["period"], len(facts["events"]))
        log.debug("facts: %s", json.dumps(facts, separators=(",", ":")))
        return {"facts": facts}

    def _draft(self, state: NarratorState) -> dict:
        feedback = (
            NARRATE_FEEDBACK.format(problems="; ".join(state["problems"]))
            if state.get("problems")
            else ""
        )
        messages = [
            {"role": "system", "content": NARRATOR_SYSTEM},
            {
                "role": "user",
                "content": NARRATE.format(facts=as_json(state["facts"]), feedback=feedback),
            },
        ]
        res = self.router.complete(self.name, messages, context={"facts": state["facts"]})
        log.info(
            "draft [%s] attempt %d: %d chars", res.route.label, state["attempt"] + 1, len(res.text)
        )
        return {
            "draft": res.text.strip(),
            "attempt": state["attempt"] + 1,
            "route": res.route.label,
        }

    def _check(self, state: NarratorState) -> dict:
        problems = check_entry(state["draft"], state["facts"])
        if problems:
            log.warning("check: draft failed fact check: %s", "; ".join(problems))
        else:
            log.info("check: draft passes fact check")
        return {"problems": problems}

    def _after_check(self, state: NarratorState) -> str:
        return "draft" if state["problems"] and state["attempt"] < self.max_attempts else "write"

    def _write(self, state: NarratorState) -> dict:
        text, route, grounded = state["draft"], state["route"], not state["problems"]
        if not grounded:
            log.warning("write: using scripted template instead of ungrounded draft")
            text = scripted.narrator(self.name, [], {"facts": state["facts"]})
            route = "scripted (fact-check fallback)"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with self.path.open("a") as fh:
            if new:
                fh.write("# Field journal\n\n")
            fh.write(f"### {state['facts']['period']}\n\n{text}\n\n*written by {route}*\n\n")
        entry = {
            "period": state["facts"]["period"],
            "text": text,
            "route": route,
            "grounded": grounded,
        }
        self.entries.append(entry)
        log.info("write: journal entry for %s (%s)", state["facts"]["period"], route)
        return {"grounded": grounded}
