"""Every validator rule fires when it should, and network intents conserve mass."""

import numpy as np
import pytest

from sim.engine import Simulation
from sim.intents import RelocateHyphae, SetTradeBias, ShuttleNutrients, Spill, apply, sellable
from sim.patches import region
from tests.conftest import small_params


@pytest.fixture
def sim():
    s = Simulation(3, small_params(world={"width": 32, "height": 32}))
    s.state.mycorrhiza.n = s.state.mycorrhiza.c / 12 * 3.0  # a big sellable N store
    s.state.mycorrhiza.p = s.state.mycorrhiza.c / 100 * 3.0
    s.ledger.t0 = s.totals()
    return s


def shuttle(**kw):
    base = dict(
        tick=0, agent="mycelium", from_patch="r0c0", to_patch="r0c1", element="n", amount_g=0.5
    )
    return ShuttleNutrients(**{**base, **kw})


def check(sim, intent, history=()):
    return sim.validator.check(intent, sim.state, history)


def test_valid_intents_pass(sim):
    assert check(sim, shuttle()) == []
    assert (
        check(
            sim,
            RelocateHyphae(
                tick=0, agent="mycelium", from_patch="r1c1", to_patch="r1c2", fraction=0.1
            ),
        )
        == []
    )
    assert check(sim, SetTradeBias(tick=0, agent="mycelium", patch="r2c2", bias=0.8)) == []


@pytest.mark.parametrize(
    ("intent", "rule"),
    [
        (shuttle(agent="narrator"), "authority"),
        (Spill(tick=0, agent="mycelium", x=3, y=3, radius=1, mass_g=1), "authority"),
        (shuttle(to_patch="r9c9"), "geometry"),
        (shuttle(to_patch="nowhere"), "geometry"),
        (shuttle(to_patch="r0c0"), "geometry"),
        (shuttle(to_patch="r3c3"), "network"),  # 6 hops
        (
            RelocateHyphae(
                tick=0, agent="mycelium", from_patch="r0c0", to_patch="r0c2", fraction=0.1
            ),
            "network",
        ),
        (shuttle(amount_g=5000.0), "resources"),
        (
            RelocateHyphae(
                tick=0, agent="mycelium", from_patch="r0c0", to_patch="r0c1", fraction=0.5
            ),
            "bounds",
        ),
        (SetTradeBias(tick=0, agent="mycelium", patch="r0c0", bias=0.6), "bounds"),  # step
        (SetTradeBias(tick=0, agent="mycelium", patch="r0c0", bias=3.0), "bounds"),  # range
    ],
)
def test_rules_fire(sim, intent, rule):
    violations = check(sim, intent)
    assert any(v.startswith(rule) for v in violations), violations


def test_network_rule_needs_hyphae(sim):
    sim.state.mycorrhiza.c[region("r0c1", sim.state.shape, 8)] = 0.0
    assert any(v.startswith("network") for v in check(sim, shuttle()))


def test_rate_limits(sim):
    history = [shuttle(to_patch=f"r{r}c0", from_patch="r1c1") for r in (0, 2, 3)]
    history.append(SetTradeBias(tick=0, agent="mycelium", patch="r3c3", bias=0.9))
    assert any(
        v.startswith("rate") and "already has 4" in v for v in check(sim, shuttle(), history)
    )
    recent = [shuttle(tick=0)]
    later = shuttle(tick=100)  # cooldown is 336 ticks
    assert any("cooling down" in v for v in check(sim, later, recent))
    assert not any("cooling down" in v for v in check(sim, shuttle(tick=400), recent))


def test_user_is_not_rate_limited(sim):
    history = [shuttle(agent="user", to_patch=f"r{r}c0", from_patch="r1c1") for r in (0, 2, 3)]
    history += [shuttle(agent="user")] * 2
    assert check(sim, shuttle(agent="user"), history) == []


def test_all_violations_reported_at_once(sim):
    bad = ShuttleNutrients(
        tick=0, agent="narrator", from_patch="r0c0", to_patch="r3c3", element="n", amount_g=5000.0
    )
    rules = {v.split(":")[0] for v in check(sim, bad)}
    assert {"authority", "network", "resources"} <= rules


@pytest.mark.parametrize("element", ["n", "p"])
def test_shuttle_moves_store_and_books_cost(sim, element):
    s = sim.state
    src, dst = region("r0c0", s.shape, 8), region("r0c1", s.shape, 8)
    before_src = getattr(s.mycorrhiza, element)[src].sum()
    before_dst = getattr(s.mycorrhiza, element)[dst].sum()
    amount = 0.3 * float(sellable(s, element, sim.params)[src].sum())
    apply(shuttle(element=element, amount_g=amount), s, sim.ledger, sim.params)
    assert getattr(s.mycorrhiza, element)[src].sum() == pytest.approx(before_src - amount)
    assert getattr(s.mycorrhiza, element)[dst].sum() == pytest.approx(before_dst + amount)
    assert sim.ledger.flows("c")["resp_network_transport"] < 0
    sim.ledger.check(sim.totals(), s.atmosphere_c, 0)  # balances still close


def test_relocate_and_bias_conserve_mass(sim):
    s = sim.state
    apply(
        RelocateHyphae(tick=0, agent="mycelium", from_patch="r1c1", to_patch="r1c2", fraction=0.2),
        s,
        sim.ledger,
        sim.params,
    )
    apply(
        SetTradeBias(tick=0, agent="mycelium", patch="r2c2", bias=0.75), s, sim.ledger, sim.params
    )
    sim.ledger.check(sim.totals(), s.atmosphere_c, 0)
    assert np.all(s.trade_bias[region("r2c2", s.shape, 8)] == 0.75)
    assert s.trade_bias.sum() == pytest.approx(s.trade_bias.size - 64 * 0.25)


def test_engine_refuses_invalid_intents_at_apply_time(sim):
    sim.submit([shuttle(tick=1, amount_g=5000.0)])
    sim.run(2)
    assert not sim.accepted and len(sim.rejected) == 1
    assert "resources" in sim.rejected[0][1]
    assert any(e.kind == "intent_rejected" for e in sim.events.events)
