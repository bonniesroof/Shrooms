"""Guild behaviour: each test isolates one mechanism on a small world."""

import numpy as np
import pytest

from sim.contamination import toxicity
from sim.decomposers import grow_on
from sim.engine import Simulation
from sim.env import TickContext
from sim.insects import step_insects
from sim.mycorrhiza import step_mycorrhiza
from sim.params import ContaminationParams
from sim.weather import Weather
from sim.world import Pool
from tests.conftest import small_params


def _ctx(sim: Simulation, temp_c: float = 20.0, shortwave: float = 600.0) -> TickContext:
    """A midsummer-noon context with fixed weather, 60% moisture and clean soil."""
    wx = Weather(day_of_year=180, hour=12, temp_c=temp_c, shortwave=shortwave, rain_mm=0.0)
    shape = sim.state.shape
    return TickContext(wx, np.full(shape, 0.6), np.ones(shape), sim.ledger)


def test_lateral_flow_makes_valleys_wetter():
    sim = Simulation(3, small_params())
    sim.run(24 * 120)  # a wet spring
    e, m = sim.state.elevation.ravel(), sim.state.moisture().ravel()
    assert np.corrcoef(e, m)[0, 1] < -0.3


def test_microbes_mineralize_rich_food_and_immobilize_poor_food(small_sim):
    s, m = small_sim.state, small_sim.params.decomposers.bacteria
    ctx = _ctx(small_sim)
    shape = s.shape
    guild = Pool.from_c(np.full(shape, 10.0), m.cn, m.cp)

    n0 = s.mineral_n.copy()
    grow_on(guild, Pool.from_c(np.full(shape, 1.0), 6.0, 30.0), m, s, ctx, "bacteria")
    assert (s.mineral_n > n0).all(), "N-rich food should release mineral N"

    n1 = s.mineral_n.copy()
    grow_on(guild, Pool.from_c(np.full(shape, 1.0), 80.0, 30.0), m, s, ctx, "bacteria")
    assert (s.mineral_n < n1).all(), "N-poor food should draw down mineral N"


def test_market_no_demand_no_trade(small_sim):
    s, pp = small_sim.state, small_sim.params.plants
    s.plant.n = s.plant.c / pp.target_cn * 1.1  # well fed
    s.plant.p = s.plant.c / pp.target_cp * 1.1
    gpp = np.full(s.shape, 1.0)
    delivered, paid = step_mycorrhiza(s, gpp, small_sim.params.mycorrhiza, pp, _ctx(small_sim))
    assert paid.sum() == 0.0 and delivered["n"].sum() == 0.0


def test_market_starved_plants_buy_more(small_sim):
    s, pp, mp = small_sim.state, small_sim.params.plants, small_sim.params.mycorrhiza
    s.plant.c = np.full(s.shape, 300.0)
    s.mycorrhiza.n = s.mycorrhiza.c / mp.guild.cn * 2.0  # fungi hold a big store
    s.mycorrhiza.p = s.mycorrhiza.c / mp.guild.cp * 2.0
    gpp = np.full(s.shape, 1.0)
    paid = {}
    for label, ratio in (("mild", 0.9), ("starved", 0.4)):
        trial = Simulation(7, small_sim.params)
        ts = trial.state
        ts.plant.c, ts.mycorrhiza.n, ts.mycorrhiza.p = s.plant.c, s.mycorrhiza.n, s.mycorrhiza.p
        ts.plant.n = ts.plant.c / pp.target_cn * ratio
        ts.plant.p = ts.plant.c / pp.target_cp * ratio
        _, c = step_mycorrhiza(ts, gpp, mp, pp, _ctx(trial))
        paid[label] = c.sum()
    assert paid["starved"] > paid["mild"] > 0


def test_fungi_never_sell_below_their_own_structure(small_sim):
    s, pp, mp = small_sim.state, small_sim.params.plants, small_sim.params.mycorrhiza
    s.plant.n = s.plant.c / pp.target_cn * 0.2
    for _ in range(200):
        step_mycorrhiza(s, np.full(s.shape, 2.0), mp, pp, _ctx(small_sim))
    assert (s.mycorrhiza.n >= s.mycorrhiza.c / mp.guild.cn - 1e-9).all()
    assert (s.mycorrhiza.p >= s.mycorrhiza.c / mp.guild.cp - 1e-9).all()


def test_insects_do_not_feed_in_the_cold(small_sim):
    s = small_sim.state
    plant0 = s.plant.c.copy()
    step_insects(s, small_sim.params.insects, _ctx(small_sim, temp_c=0.0))
    np.testing.assert_array_equal(s.plant.c, plant0)


def test_toxicity_dose_response():
    p = ContaminationParams(ec50_g=10.0)
    t = toxicity(np.array([0.0, 10.0, 100.0]), p)
    assert t[0] == 1.0 and t[1] == pytest.approx(0.5) and t[2] < 0.1


def test_bacteria_degrade_contaminant():
    params = small_params()
    clean = params.contamination.model_copy(update={"hotspots": 1, "hotspot_radius": 3.0})
    sim = Simulation(4, params.model_copy(update={"contamination": clean}))
    start = sim.state.contaminant.sum()
    sim.run(24 * 150)
    assert sim.ledger.flows("contaminant")["degradation"] < 0
    assert sim.state.contaminant.sum() < start
