"""Phase 5, step 1: static graph JEPA and linear probes.

Probe and collapse-stat tests are numpy-only. Model tests need PyTorch (the
optional `worldmodel` group) and are skipped without it, as in CI.
"""

import numpy as np
import pytest

from worldmodel.dataset import generate_run
from worldmodel.probes import (
    LinearProbe,
    Ridge,
    collapse_stats,
    forecast_probes,
    probe_targets,
    r2,
    run_probes,
)
from worldmodel.targets import HORIZONS_DAYS, build_samples, scores


def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 6))
    y = x @ np.array([1.0, -2.0, 0.5, 0.0, 0.0, 3.0]) + 4.0
    assert r2(y, Ridge(1e-6).fit(x, y).predict(x)) > 0.999


def test_collapse_stats_tell_spread_from_collapse():
    rng = np.random.default_rng(0)
    spread = collapse_stats(rng.normal(size=(1000, 16)))
    collapsed = collapse_stats(np.outer(rng.normal(size=1000), np.ones(16)))  # rank one
    assert spread["effective_rank"] > 14 and collapsed["effective_rank"] < 1.01
    assert collapse_stats(np.ones((100, 16)))["std"] == 0.0


def test_probes_run_and_raw_features_probe_themselves():
    x = generate_run(903, days=40)["x"]
    t = probe_targets(x)
    assert t["biomass"].shape == (40, 64) and t["links"].shape == (40, 112)
    days = {"train": slice(0, 24), "val": slice(24, 32), "test": slice(32, 40)}
    feats = {s: np.log1p(np.abs(x[sl])) for s, sl in days.items()}
    targets = {s: probe_targets(x[sl]) for s, sl in days.items()}
    out = run_probes(feats, targets)
    assert set(out) == {"biomass", "contamination", "moisture", "network", "links", "mean_test_r2"}
    assert out["biomass"]["test_r2"] > 0.99  # log1p(plant C) is itself a raw feature


def test_lad_probe_fits_the_median_not_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    y = x @ np.array([1.0, 0.0, -1.0])
    y[:40] += 50.0  # 10% gross outliers pull least squares, not least absolute deviations
    p = LinearProbe(x, y)
    lad = np.abs(p.predict(p.lad(1e-6), x[40:]) - y[40:]).mean()
    ls = np.abs(p.predict(p.ridge(1e-6), x[40:]) - y[40:]).mean()
    assert lad < 0.05 < ls


@pytest.fixture(scope="module")
def forecast_samples():
    """Phase 4 samples from a short world, split by time into train / val / test."""
    s = build_samples(generate_run(905, days=110))  # 50 samples
    cut = {"train": slice(0, 30), "val": slice(30, 40), "test": slice(40, 50)}
    return {k: {n: v[sl] for n, v in s.items()} for k, sl in cut.items()}


def test_forecast_probes_score_like_the_gnn(forecast_samples):
    s = forecast_samples
    feats = {k: v["nodes"][:, None].astype(np.float32) for k, v in s.items()}
    pred = forecast_probes(feats, s)
    for name in ("biomass", "contamination", "moisture", "mortality", "links"):
        assert pred[name].shape == s["test"][f"y_{name}"].shape
    assert 0 <= pred["links"].min() and pred["links"].max() <= 1
    out = scores(pred, s["test"])
    assert set(out) >= {f"mean_skill@{h}d" for h in HORIZONS_DAYS}
    # combined representations are gathered piecewise, identical to concatenating first
    both = forecast_probes({k: (v, v) for k, v in feats.items()}, s)
    cat = forecast_probes({k: np.concatenate([v, v], -1) for k, v in feats.items()}, s)
    assert all(np.allclose(both[n], cat[n]) for n in both)


# --- the model ------------------------------------------------------------------------------

try:
    import torch

    from worldmodel.jepa import StaticJEPA, TemporalJEPA, block_mask, momentum_at
    from worldmodel.train_jepa import inputs, normalizers, train
    from worldmodel.train_jepa_temporal import Worlds, samples_for
    from worldmodel.train_jepa_temporal import train as train_temporal
except ImportError:  # the optional `worldmodel` group isn't installed (CI)
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="needs the worldmodel group (PyTorch)")


@pytest.fixture(scope="module")
def tiny():
    x = generate_run(904, days=48)
    snaps = {"x": x["x"], "g": x["g"]}
    meta = normalizers(snaps)
    return inputs(snaps, meta)


def make(seed=0):
    torch.manual_seed(seed)
    return StaticJEPA(n_in=14, n_glob=7, hidden=16, layers=2, pred_layers=1)


@needs_torch
def test_shapes(tiny):
    m = make()
    b = {k: v[:5] for k, v in tiny.items()}
    hidden = block_mask(5, 8, 8, 0.3, torch.Generator().manual_seed(0))
    assert m.encode(b["nodes"], b["glob"], b["on"], b["elev"]).shape == (5, 64, 16)
    ctx, pred = m.predict(b["nodes"], b["glob"], b["on"], b["elev"], hidden)
    assert ctx.shape == pred.shape == (5, 64, 16)
    out = m.loss(b["nodes"], b["glob"], b["on"], b["elev"], hidden)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])


@needs_torch
def test_block_mask_hides_the_requested_share_in_blocks():
    gen = torch.Generator().manual_seed(1)
    h = block_mask(32, 8, 8, 0.3, gen)
    counts = h.sum(1)
    assert (counts >= 19).all() and (counts <= 19 + 8).all()  # >= 30% of 64, overshoot < 1 block
    grid = h.reshape(32, 8, 8)
    lonely = grid[:, 1:-1, 1:-1] & ~grid[:, :-2, 1:-1] & ~grid[:, 2:, 1:-1] \
        & ~grid[:, 1:-1, :-2] & ~grid[:, 1:-1, 2:]  # fmt: skip
    assert not lonely.any()  # every hidden patch belongs to a block
    assert torch.equal(h, block_mask(32, 8, 8, 0.3, torch.Generator().manual_seed(1)))


@needs_torch
def test_masked_patches_cannot_leak_into_the_context(tiny):
    m = make()
    b = {k: v[:2].clone() for k, v in tiny.items()}
    hidden = torch.zeros(2, 64, dtype=torch.bool)
    hidden[:, 27] = True
    _, before = m.predict(b["nodes"], b["glob"], b["on"], b["elev"], hidden)
    b["nodes"][:, 27] += 5.0  # change everything about the hidden patch except terrain
    b["on"][:, 27] = 1 - b["on"][:, 27]
    _, after = m.predict(b["nodes"], b["glob"], b["on"], b["elev"], hidden)
    assert torch.allclose(before, after)


@needs_torch
def test_ema_moves_the_target_toward_the_context_and_gets_no_grad(tiny):
    m = make()
    assert all(not p.requires_grad for p in m.target.parameters())
    with torch.no_grad():
        for p in m.context.parameters():
            p.add_(1.0)
    pc = next(m.context.parameters())
    pt = next(m.target.parameters())
    gap = (pc - pt).abs().mean()
    m.ema_update(0.9)
    assert torch.allclose((pc - pt).abs().mean(), 0.9 * gap, rtol=1e-4)
    m.ema_update(1.0)  # momentum 1 freezes the target
    assert torch.allclose((pc - pt).abs().mean(), 0.9 * gap, rtol=1e-4)
    assert momentum_at(0, 100) == pytest.approx(0.996) and momentum_at(100, 100) == 1.0


@needs_torch
def test_loss_decreases_on_a_tiny_run(tiny):
    m = make()
    curve = train(m, tiny, epochs=6, batch=8, lr=3e-3, mask_ratio=0.3, var_weight=0.1, seed=0,
                  log=lambda s: None)  # fmt: skip
    assert curve[-1]["jepa"] < 0.7 * curve[0]["jepa"], [c["jepa"] for c in curve]
    assert curve[-1]["target_rank"] > 2  # not collapsed to a point or a line


@needs_torch
def test_training_is_deterministic_with_a_seed(tiny):
    runs = []
    for _ in range(2):
        m = make(seed=3)
        curve = train(m, tiny, epochs=2, batch=8, lr=1e-3, mask_ratio=0.3, var_weight=0.1,
                      seed=3, log=lambda s: None)  # fmt: skip
        runs.append((curve[-1]["loss"], next(m.target.parameters()).clone()))
    assert runs[0][0] == runs[1][0] and torch.equal(runs[0][1], runs[1][1])


# --- temporal JEPA ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_worlds(tmp_path_factory):
    d = tmp_path_factory.mktemp("worlds")
    run = generate_run(906, days=75)
    np.savez(d / "run_906.npz", **run)
    meta = normalizers({"x": run["x"], "g": run["g"]})
    return Worlds([906], d, meta), meta


def make_temporal(seed=0):
    torch.manual_seed(seed)
    return TemporalJEPA(n_in=14, n_glob=7, hidden=16, layers=2, pred_layers=1)


@needs_torch
def test_worlds_line_up_with_the_phase4_samples(tiny_worlds):
    w, _ = tiny_worlds
    s = samples_for(w, stride=1)
    assert len(w.t) == len(s["day"]) == 75 - 30 - 30
    assert np.array_equal(w.t.numpy(), s["day"])  # one world: flat index == day
    later = w.at(w.t, HORIZONS_DAYS[-1])["elev"]
    assert later.shape == (len(w.t), 64)


@needs_torch
def test_temporal_shapes_and_horizon_conditioning(tiny_worlds):
    w, _ = tiny_worlds
    m = make_temporal()
    with torch.no_grad():
        m.predictor.horizon.weight.normal_()  # distinct horizon embeddings
    rows = w.t[:4]
    now, past = w.at(rows), w.at(rows, -7)
    z, p0 = m.predict(now, past, torch.zeros(4, dtype=torch.long))
    _, p2 = m.predict(now, past, torch.full((4,), 2))
    assert z.shape == p0.shape == (4, 64, 16)
    assert not torch.allclose(p0, p2)
    out = m.loss(now, past, [w.at(rows, h) for h in HORIZONS_DAYS])
    assert out["per_horizon"].shape == (3,) and torch.isfinite(out["loss"])
    assert all(not p.requires_grad for p in m.target.parameters())


@needs_torch
def test_temporal_encoders_start_from_a_static_checkpoint():
    torch.manual_seed(1)
    static = StaticJEPA(n_in=14, n_glob=7, hidden=16, layers=2, pred_layers=1)
    m = make_temporal()
    m.init_encoders(static.state_dict())
    for name in ("context", "target"):
        a, b = getattr(m, name).state_dict(), getattr(static, name).state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)


@needs_torch
def test_temporal_loss_decreases_and_is_deterministic(tiny_worlds):
    w, _ = tiny_worlds
    curves = []
    for _ in range(2):
        m = make_temporal(seed=2)
        curves.append(train_temporal(m, w, epochs=5, batch=5, lr=3e-3, var_weight=0.1, seed=2,
                                     log=lambda s: None))  # fmt: skip
    first, last = curves[0][0]["jepa"], curves[0][-1]["jepa"]
    assert last < 0.7 * first, [c["jepa"] for c in curves[0]]
    assert curves[0][-1]["loss"] == curves[1][-1]["loss"]
    assert curves[0][-1]["pred_rank"] > 2
