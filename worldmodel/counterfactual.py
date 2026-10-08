"""Counterfactual pairs: the same world at the same moment, with and without a spill.

For each held-out world, the research-mode run is replayed exactly as the dataset
generated it (forcing mode) up to a branch day t. The sim is then copied:

    base    30 more days with no interventions
    spill   the same, plus one random spill applied at the start of day t + 1

The sim is deterministic and the spill draws no randomness, so the branches see
the same weather and differ only by the spill: the difference between them is
the spill's true effect. Snapshots are kept at t - 7, t, and t + k for every
horizon in targets.HORIZONS_DAYS.

    uv run python -m worldmodel.counterfactual    # ~3 min, 2 cores; data/phase5/counterfactuals.npz
"""

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sim.intents import Spill
from sim.rng import stream
from worldmodel.dataset import OUT_FORCING, advance_day, daily_weather, new_sim
from worldmodel.graph import snapshot
from worldmodel.targets import HORIZONS_DAYS, TREND_DAYS

DEFAULT_PATH = OUT_FORCING / "counterfactuals.npz"
TEST_SEEDS = (24, 25, 26, 27)  # train.SPLITS["test"]
BRANCH_DAYS = tuple(range(50, 700, 50))  # 13 branch points per world


def random_spill(rng: np.random.Generator, tick: int) -> Spill:
    """Same ranges as dataset.random_intent's spills."""
    return Spill(tick=tick, x=float(rng.uniform(4, 60)), y=float(rng.uniform(4, 60)),
                 radius=float(rng.uniform(2, 6)), mass_g=float(rng.uniform(300, 3000)))  # fmt: skip


def rollout(sim, days: int, spill: Spill | None) -> dict:
    """Run a copy of `sim` for `days` days; keep snapshots at each horizon and daily weather."""
    sim = copy.deepcopy(sim)
    n_before = len(sim.accepted)
    if spill is not None:
        sim.submit([spill])
    x, g, plant, w = [], [], [], []
    for day in range(1, days + 1):
        sim.run_until(sim.state.tick + 24)
        w.append(daily_weather(sim))
        if day in HORIZONS_DAYS:
            snap = snapshot(sim)
            x.append(snap.x)
            g.append(snap.g)
            plant.append(sim.state.plant.c.astype(np.float32))
    applied = len(sim.accepted) - n_before
    assert applied == (spill is not None), "the spill was rejected"
    return {"x": np.array(x), "g": np.array(g), "plant": np.array(plant), "w": np.array(w)}


def world_pairs(seed: int, branch_days=BRANCH_DAYS, horizon: int = max(HORIZONS_DAYS)) -> dict:
    """Every counterfactual pair from one world, replaying its dataset run up to each branch."""
    sim = new_sim(seed)
    rng = stream(seed, "dataset")  # the dataset's own intervention stream: same world
    spill_rng = stream(seed, "counterfactual")
    xs, gs = {}, {}
    keys = ("x_past", "x_now", "g_past", "g_now", "plant_now", "x_base", "x_spill", "g_fut",
            "plant_base", "plant_spill", "w", "spill", "world", "day")  # fmt: skip
    out: dict[str, list] = {k: [] for k in keys}
    for day in range(max(branch_days) + 1):
        advance_day(sim, rng, day, "forcing")
        snap = snapshot(sim)
        xs[day], gs[day] = snap.x, snap.g
        if day not in branch_days:
            continue
        spill = random_spill(spill_rng, sim.state.tick)
        base, spilled = rollout(sim, horizon, None), rollout(sim, horizon, spill)
        assert np.array_equal(base["w"], spilled["w"])  # same weather in both branches
        out["x_past"].append(xs[day - TREND_DAYS])
        out["g_past"].append(gs[day - TREND_DAYS])
        out["x_now"].append(snap.x)
        out["g_now"].append(snap.g)
        out["plant_now"].append(sim.state.plant.c.astype(np.float32))
        out["x_base"].append(base["x"])
        out["x_spill"].append(spilled["x"])
        out["g_fut"].append(base["g"])
        out["plant_base"].append(base["plant"])
        out["plant_spill"].append(spilled["plant"])
        out["w"].append(base["w"])
        out["spill"].append(json.dumps(spill.model_dump(exclude={"rationale"})))
        out["world"].append(seed)
        out["day"].append(day)
    return {k: np.array(v) for k, v in out.items()}


def generate(seeds=TEST_SEEDS, path: Path = DEFAULT_PATH, workers: int = 2,
             branch_days=BRANCH_DAYS) -> dict:  # fmt: skip
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(world_pairs, seeds, [branch_days] * len(seeds)))
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"{len(data['day'])} counterfactual pairs from {len(seeds)} worlds "
          f"in {time.perf_counter() - t0:.0f}s -> {path}", flush=True)  # fmt: skip
    return data


def load(path: Path = DEFAULT_PATH) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args()
    generate(path=args.out, workers=args.workers)


if __name__ == "__main__":
    main()
