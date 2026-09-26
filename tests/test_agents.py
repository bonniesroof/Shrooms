"""Phase 2 done-when: the keystone agent makes validated decisions that change the
sim, and replay reproduces them without calling the LLM. Plus the graph loop,
checkpointing and the Narrator's fact check."""

import json
import sqlite3

import pytest

from cognition import scripted
from cognition.keystone import MyceliumAgent
from cognition.llm import LLMResult, ModelRoute, Router
from cognition.narrator import Narrator, check_entry
from cognition.runner import run_with_agents
from sim.engine import Simulation
from sim.replay import replay
from tests.conftest import small_params

PARAMS = small_params(world={"width": 32, "height": 32})
DAYS = 120  # spring into summer, so the market is active
TICKS = DAYS * 24


def scripted_router() -> Router:
    return Router(
        {r: [ModelRoute(backend="scripted")] for r in ("mycelium", "narrator")},
        scripted=scripted.POLICIES,
    )


@pytest.fixture(scope="module")
def agent_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("agents")
    return run_with_agents(5, TICKS, scripted_router(), out, "t", params=PARAMS)


def test_agent_decisions_are_validated_and_applied(agent_run):
    sim = agent_run.sim
    mine = [i for i in sim.accepted if i.agent == "mycelium"]
    assert len(agent_run.mycelium.traces) == TICKS // 168
    assert mine, "the keystone agent never got an intent applied"
    assert {i.kind for i in mine} >= {"shuttle_nutrients", "set_trade_bias"}
    assert not sim.rejected  # everything it committed survived the engine's re-check


def test_agent_changes_the_sim(agent_run):
    baseline = Simulation(5, PARAMS)
    baseline.run(TICKS)
    assert baseline.state_hash() != agent_run.sim.state_hash()
    assert (agent_run.sim.state.trade_bias != 1.0).any()


def test_replay_reproduces_agent_run_without_llm(agent_run, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("replay must never call a model")

    monkeypatch.setattr(Router, "complete", boom)
    replayed = replay(agent_run.record)
    assert replayed.state_hash() == agent_run.record.final_hash
    assert [i.kind for i in replayed.accepted] == [i.kind for i in agent_run.record.intents]


def test_artifacts_written(agent_run):
    p = agent_run.paths
    traces = [json.loads(line) for line in p["trace"].read_text().splitlines()]
    assert len(traces) == len(agent_run.mycelium.traces)
    assert {"observation", "deliberation", "accepted", "rejected"} <= set(traces[0])
    journal = p["journal"].read_text()
    assert journal.startswith("# Field journal") and journal.count("### ") == DAYS // 30


def test_sqlite_checkpointer_holds_both_agent_threads(agent_run):
    conn = sqlite3.connect(agent_run.paths["checkpoints"])
    threads = {r[0] for r in conn.execute("select distinct thread_id from checkpoints")}
    assert threads == {"t:mycelium", "t:narrator"}
    n = conn.execute("select count(*) from checkpoints where thread_id='t:mycelium'").fetchone()
    assert n[0] >= 5 * len(agent_run.mycelium.traces)  # one per node per decision, at least


class FakeModel:
    """Proposes an invalid shuttle first, then a valid bias change after feedback."""

    def __init__(self):
        self.prompts = []

    def __call__(self, role, messages, context):
        self.prompts.append(messages[-1]["content"])
        if context["step"] == "deliberate":
            return "I will move nitrogen."
        if len([p for p in self.prompts if "Reply with ONLY" in p]) == 1:
            return json.dumps(
                {
                    "intents": [
                        {
                            "kind": "shuttle_nutrients",
                            "from_patch": "r0c0",
                            "to_patch": "r3c3",
                            "element": "n",
                            "amount_g": 9999,
                            "rationale": "greedy",
                        },
                        {"kind": "teleport", "rationale": "not a real action"},
                    ]
                }
            )
        return json.dumps(
            {
                "intents": [
                    {"kind": "set_trade_bias", "patch": "r1c1", "bias": 0.9, "rationale": "invest"}
                ]
            }
        )


def test_validator_feedback_loop():
    model = FakeModel()
    router = Router({"mycelium": [ModelRoute(backend="scripted")]}, scripted={"mycelium": model})
    agent = MyceliumAgent(router)
    sim = Simulation(5, PARAMS)
    sim.run(24)
    committed = agent.decide(sim)
    trace = agent.traces[-1]
    assert [i.kind for i in committed] == ["set_trade_bias"]
    assert trace["attempt"] == 2 and len(trace["rejected"]) == 2
    rules = {v.split(":")[0] for r in trace["rejected"] for v in r["violations"]}
    assert {"schema", "network", "resources"} <= rules
    assert "REJECTED" in model.prompts[2]  # the retry saw the validator's feedback


def test_fact_check_catches_invented_numbers():
    facts = {"period": "Y1 Jun", "rain_mm": 42, "per_m2_now": {"plant_c_g": 480}}
    assert check_entry("Rain totalled 42 mm; plants hold 480 g C.", facts) == []
    problems = check_entry("Rain totalled 42 mm and 17 new beetle species arrived.", facts)
    assert problems == ["number 17 is not in the facts"]


def test_narrator_falls_back_when_model_keeps_inventing(tmp_path):
    router = Router(
        {"narrator": [ModelRoute(backend="scripted")]},
        scripted={"narrator": lambda *a: "Exactly 9999 fireflies appeared."},
    )
    narr = Narrator(router, tmp_path / "j.md")
    sim = Simulation(5, PARAMS)
    sim.run(24 * 30)
    narr.narrate(sim)
    entry = narr.entries[-1]
    assert not entry["grounded"] and entry["route"].startswith("scripted (fact-check")
    assert "9999" not in (tmp_path / "j.md").read_text()


def test_llm_result_is_plain_data():
    assert LLMResult("x", ModelRoute(backend="scripted"), 0.0, 1, False).text == "x"
