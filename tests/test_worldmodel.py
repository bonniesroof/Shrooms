"""Phase 4: transport kernel, graph builder, targets, dataset, and the forecaster gate.

Done-when: the GNN beats persistence at 7 and 30 days (test_gate_*), and
forecasts are visible in-game (test_server.py::test_frame_carries_forecasts).
The gate runs the shipped, numpy-exported model on a world it never saw.
"""

import numpy as np
import pytest

from sim.engine import Simulation
from sim.transport import route, route_reference, uniform_fractions
from worldmodel.dataset import generate_run
from worldmodel.forecast import DEFAULT_PATH, NumpyForecaster
from worldmodel.graph import EDGE_TYPES, adjacent_pairs, dense_adjacency, edges, snapshot
from worldmodel.targets import HORIZONS_DAYS, build_samples, mortality, scores


@pytest.mark.parametrize(
    ("shape", "frac_shape"),
    [((64, 64), (4, 64, 64)), ((3, 64, 64), (3, 4, 64, 64)), ((3, 64, 64), (4, 64, 64))],
)
def test_compiled_route_is_bit_identical_to_reference(shape, frac_shape):
    rng = np.random.default_rng(0)
    a, f = rng.gamma(2, 100, size=shape), rng.uniform(0, 0.24, size=frac_shape)
    assert np.array_equal(route(a, f), route_reference(a, f))
    shared = uniform_fractions(shape[-2:], 0.3)
    assert np.array_equal(route(a, shared), route_reference(a, shared))


def test_graph_has_typed_nodes_and_edges():
    sim = Simulation(3)
    sim.run(24 * 10)
    snap = snapshot(sim)
    assert snap.x.shape == (64, 14) and snap.g.shape == (7,)
    e = edges(snap.x, 8, 8)
    assert set(e) == set(EDGE_TYPES)
    assert e["adjacent"][0].shape[1] == 2 * len(adjacent_pairs(8, 8)) == 224
    src, dst = e["downslope"][0]
    elev = snap.x[:, 12]
    assert (elev[src] > elev[dst]).all()  # downslope edges only go downhill
    a = dense_adjacency(snap.x, 8, 8)
    assert np.allclose(a[0].sum(axis=1), 1.0)  # adjacent rows are normalized


def test_hyphal_edges_follow_the_fungal_network():
    sim = Simulation(3)
    sim.state.mycorrhiza.c[:, :] = 0.0  # no network anywhere
    snap = snapshot(sim)
    assert edges(snap.x, 8, 8)["hyphal"][0].shape[1] == 0


def test_mortality_counts_vegetated_cells_that_die_back():
    before = np.full((16, 16), 100.0)
    after = before.copy()
    after[:4, :4] = 10.0  # 16 of the 64 cells in patch r0c0 die back
    m = mortality(before, after, size=8)
    assert m[0] == pytest.approx(0.25) and m[1:].max() == 0.0


def test_dataset_run_and_samples_are_deterministic():
    a, b = generate_run(901, days=75), generate_run(901, days=75)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["plant"], b["plant"])
    s = build_samples(a)
    assert s["nodes"].shape[1:] == (64, 28)
    assert s["y_links"].shape[1:] == (112, len(HORIZONS_DAYS))
    assert len(s["day"]) == 75 - 30 - 30  # needs 30 days behind and ahead


def test_persistence_has_zero_skill_against_itself():
    run = generate_run(902, days=75)
    s = build_samples(run)
    pred = {n: s[f"p_{n}"].astype(float) for n in
            ("biomass", "contamination", "moisture", "mortality", "links")}  # fmt: skip
    out = scores(pred, s)
    assert all(abs(v["skill"]) < 1e-12 for v in out.values() if isinstance(v, dict))


def test_forecaster_runs_in_numpy_on_a_live_world():
    model = NumpyForecaster()
    sim = Simulation(5)
    sim.run(24 * 8)
    week_ago = snapshot(sim).x
    sim.run(24 * 7)
    now = snapshot(sim)
    out = model.forecast_one(now.x, week_ago, now.g)
    assert set(out) == {"biomass", "contamination", "moisture", "mortality", "links"}
    assert out["biomass"][30].shape == (64,) and out["links"][7].shape == (112,)
    assert all(np.isfinite(v).all() for head in out.values() for v in head.values())


@pytest.fixture(scope="module")
def held_out():
    """A world no training or selection run used: seed 1000, one year."""
    return build_samples(generate_run(1000, days=365))


@pytest.mark.slow
@pytest.mark.parametrize("horizon", [7, 30])
def test_gate_gnn_beats_persistence(held_out, horizon):
    model = NumpyForecaster(DEFAULT_PATH)
    out = scores(_predict_samples(model, held_out), held_out)
    assert out[f"mean_skill@{horizon}d"] > 0, {k: v for k, v in out.items() if f"@{horizon}d" in k}


def _predict_samples(model: NumpyForecaster, s: dict) -> dict:
    """Run the numpy model on prepared samples (inputs already built from raw features)."""
    m = model.meta
    nodes = (s["nodes"] - m["node_mean"]) / m["node_std"]
    glob = (s["glob"] - m["glob_mean"]) / m["glob_std"]
    return model.decode(*model.raw(nodes, glob, s["on"], s["elev"], s["link_now"].astype(float)))
