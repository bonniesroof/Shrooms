"""Phase 3 server: tools and budget, modes, director, whisper, replay, protocol.

Gate B (playable end to end) is covered by test_brownfield_is_winnable_*.
"""

import base64
import dataclasses

import pytest
from fastapi.testclient import TestClient

from cognition.llm import Router
from server import players
from server.app import ServerConfig, create_app
from server.scenarios import Brownfield
from server.session import TOOLS, GameSession
from sim import TICKS_PER_DAY
from sim.replay import replay


def test_tools_cost_budget_in_game_mode():
    g = GameSession(seed=1, agents=False)
    start = g.budget
    assert g.use_tool("compost", 30, 30)["ok"]
    assert g.budget == start - TOOLS["compost"].cost
    g.budget = 5
    out = g.use_tool("excavate", 30, 30)
    assert not out["ok"] and "costs" in out["error"]


def test_research_mode_is_free_and_unlocks_spill():
    g = GameSession(seed=1, agents=False, mode="game")
    assert "research mode" in g.use_tool("spill", 10, 10)["error"]
    g.set_mode("research")
    budget = g.budget
    assert g.use_tool("spill", 10, 10)["ok"] and g.use_tool("excavate", 30, 30)["ok"]
    assert g.budget == budget and g.research_used


def test_tool_validation_errors_reach_the_player():
    g = GameSession(seed=1, agents=False)
    out = g.use_tool("irrigate", 99, 10)
    assert not out["ok"] and out["error"].startswith("geometry")


def test_scenario_setup_strips_and_contaminates_the_site():
    g = GameSession(seed=1, agents=False)
    g.advance(TICKS_PER_DAY)
    assert not g.sim.rejected
    kinds = [i.kind for i in g.sim.accepted if i.agent == "scenario"]
    assert kinds.count("excavate") == 4 and kinds.count("spill") == 4
    rows = {o["key"]: o for o in g.status["objectives"]}
    assert rows["contaminant_left"]["value"] == pytest.approx(1.0)
    assert rows["network_patches"]["value"] == 0


def _director_intents(mode: str) -> list:
    g = GameSession(seed=42, agents=False, mode=mode)
    g.advance(170 * TICKS_PER_DAY)  # the seed-42 director first acts before day 170
    return [i for i in g.sim.accepted if i.agent == "director"]


def test_director_acts_in_game_mode_only():
    assert _director_intents("game")
    assert not _director_intents("research")


def test_whisper_reaches_the_keystone_and_the_inspector():
    g = GameSession(seed=42)
    g.advance(24 * 150)  # into the growing season, so the market is open
    g.whisper("please help r2c3")
    g.advance(168)
    view = g.inspector()
    assert view["whisper"] == "please help r2c3"
    assert "gardener whispered" in view["deliberation"]
    assert any(m["who"] == "you" and "whispered" in m["text"] for m in g.feed)


def test_game_session_replays_without_calling_a_model(monkeypatch):
    g = GameSession(seed=5)
    players.play(g, players.greedy, max_days=120)
    record = g.record()
    kinds = {i.agent for i in record.intents}
    assert {"scenario", "user", "mycelium"} <= kinds

    def boom(*a, **k):
        raise AssertionError("replay must not call a model")

    monkeypatch.setattr(Router, "complete", boom)
    assert replay(record).state_hash() == record.final_hash


def test_frame_fields_decode_to_the_grid():
    g = GameSession(seed=1, agents=False)
    f = g.frame()
    h, w = f["grid"]
    for name, data in f["fields"].items():
        assert len(base64.b64decode(data)) == h * w, name
    assert f["scenario"]["status"] == "playing" and f["budget"] == g.budget


def test_losing_at_the_deadline():
    g = GameSession(seed=1, agents=False)
    g.scenario = dataclasses.replace(g.scenario, deadline_days=20)
    assert players.play(g, players.idle)["status"] == "lost"
    assert g.status["day"] == 20
    assert not g.use_tool("compost", 30, 30)["ok"]  # the game is over


@pytest.mark.slow
def test_brownfield_is_winnable_by_a_sensible_player():
    """Gate B, playable end to end: a scripted gardener wins before the deadline."""
    g = GameSession(seed=42)
    status = players.play(g, players.greedy)
    assert status["status"] == "won", status["objectives"]
    assert status["day"] < Brownfield().deadline_days


def _reply(ws) -> dict:
    """Next non-frame message: the server broadcasts frames between replies."""
    while (msg := ws.receive_json())["type"] == "frame":
        pass
    return msg


def test_websocket_protocol():
    with TestClient(create_app(ServerConfig(seed=1))) as client:
        assert client.get("/api/health").json()["ok"]
        with client.websocket_connect("/ws") as ws:
            hello, first = ws.receive_json(), ws.receive_json()
            assert hello["type"] == "hello" and "compost" in hello["tools"]
            assert first["type"] == "frame" and first["tick"] == 0
            ws.send_json({"cmd": "step", "days": 2})
            assert _reply(ws) == {"type": "ack", "cmd": "step"}
            assert client.app.state.hub.session.sim.state.tick == 48
            ws.send_json({"cmd": "tool", "tool": "seed", "x": 20, "y": 20})
            assert _reply(ws)["type"] == "ack"
            ws.send_json({"cmd": "speed", "value": 7})
            assert _reply(ws)["type"] == "error"
            ws.send_json({"cmd": "tool", "tool": "nope", "x": 1, "y": 1})
            assert "unknown tool" in _reply(ws)["error"]
            ws.send_json({"cmd": "step", "days": 1})  # queued tools apply on the next tick
            assert _reply(ws)["type"] == "ack"
        record = client.get("/api/record").json()
        assert any(i["kind"] == "seed" for i in record["intents"])
