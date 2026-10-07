"""Phase 5, step 1: static graph JEPA and linear probes.

Probe and collapse-stat tests are numpy-only. Model tests need PyTorch (the
optional `worldmodel` group) and are skipped without it, as in CI.
"""

import numpy as np
import pytest

from worldmodel.dataset import generate_run
from worldmodel.probes import Ridge, collapse_stats, probe_targets, r2, run_probes


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


# --- the model ------------------------------------------------------------------------------

try:
    import torch

    from worldmodel.jepa import StaticJEPA, block_mask, momentum_at
    from worldmodel.train_jepa import inputs, normalizers, train
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
