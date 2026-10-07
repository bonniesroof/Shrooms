"""Research-mode dataset: batched headless runs, snapshotted daily.

Each run is one world (a meadow, or the brownfield site) under seeded random
interventions, so the data covers recovery, contamination and disturbance as
well as quiet seasons. Research mode: no director, no LLM agents. Runs execute
in parallel worker processes ("batched"); throughput is reported as sim ticks
per second across all workers.

Modes:
    phase4    an intervention on day 15 of each 30, with probability 0.6 (the Phase 4 data)
    forcing   an intervention on day 5 of each 10, with probability 0.6, spills more
              likely; the applied intents are recorded, so a model can be conditioned
              on them (Phase 5, action-conditioned JEPA). Written to data/phase5.

Per run, data/<phase>/run_<seed>.npz holds:
    x       (days, nodes, features)  patch features (graph.PATCH_FEATURES), raw units
    g       (days, globals)          atmosphere-node features (graph.GLOBAL_FEATURES)
    plant   (days, H, W) float32     cell plant carbon, for the mortality target
    ticks   (days,)                  tick of each snapshot
    seed, scenario, grid (rows, cols)
    intents (forcing mode) JSON list of the user intents the sim applied, with their ticks
    w       (forcing mode) (days, 3) that day's mean temperature, total rain, mean shortwave
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sim import TICKS_PER_DAY
from sim.engine import Simulation
from sim.intents import Amend, Disturb, Excavate, Inoculate, Irrigate, Seed, Spill
from sim.params import SimParams
from sim.rng import stream
from worldmodel.graph import snapshot

OUT = Path("data/phase4")
OUT_FORCING = Path("data/phase5")
INTENT_KINDS = ("spill", "disturb", "inoculate", "compost", "seed", "irrigate", "excavate")
MODES = {  # (period in days, day within the period, probability, kind probabilities)
    "phase4": (30, 15, 0.6, (0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1)),
    "forcing": (10, 5, 0.6, (0.35, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1)),
}


def scenario_for(seed: int) -> str:
    return "brownfield" if seed % 3 == 0 else "meadow"


def random_intent(rng: np.random.Generator, tick: int, p=MODES["phase4"][3]):
    """One plausible intervention somewhere on the 64x64 grid."""
    x, y = float(rng.uniform(4, 60)), float(rng.uniform(4, 60))
    kind = rng.choice(list(INTENT_KINDS), p=list(p))
    if kind == "spill":
        return Spill(tick=tick, x=x, y=y, radius=float(rng.uniform(2, 6)),
                     mass_g=float(rng.uniform(300, 3000)))  # fmt: skip
    if kind in ("disturb", "excavate"):
        half = int(rng.integers(3, 9))
        box = {"x0": max(0, int(x) - half), "y0": max(0, int(y) - half),
               "x1": min(64, int(x) + half), "y1": min(64, int(y) + half)}  # fmt: skip
        if kind == "disturb":
            return Disturb(tick=tick, fraction=float(rng.uniform(0.3, 1.0)), **box)
        return Excavate(tick=tick, fraction=float(rng.uniform(0.3, 0.9)), **box)
    if kind == "inoculate":
        guild = str(rng.choice(["bacteria", "saprotrophs", "mycorrhiza"]))
        return Inoculate(tick=tick, x=x, y=y, radius=4, guild=guild,
                         mass_c_g=float(rng.uniform(100, 400)))  # fmt: skip
    if kind == "compost":
        return Amend(tick=tick, x=x, y=y, radius=5, mass_c_g=float(rng.uniform(1000, 5000)))
    if kind == "seed":
        return Seed(tick=tick, x=x, y=y, radius=5, mass_c_g=float(rng.uniform(200, 1000)))
    return Irrigate(tick=tick, x=x, y=y, radius=5, water_mm=float(rng.uniform(10, 40)))


def new_sim(seed: int) -> Simulation:
    """A research-mode world for this seed: a meadow, or the brownfield site after setup."""
    from server.scenarios import Brownfield  # scenario definitions live with the game

    if scenario_for(seed) == "brownfield":
        bf = Brownfield()
        sim = Simulation(seed, bf.params())
        sim.submit(bf.setup_intents())
        return sim
    return Simulation(seed, SimParams())


def advance_day(sim: Simulation, rng: np.random.Generator, day: int, mode: str) -> None:
    """One dataset day: maybe draw an intervention, then run 24 ticks."""
    period, when, prob, kinds = MODES[mode]
    if day % period == when and rng.random() < prob:
        sim.submit([random_intent(rng, sim.state.tick, kinds)])
    sim.run_until(sim.state.tick + TICKS_PER_DAY)


def daily_weather(sim: Simulation) -> list[float]:
    """The last day's mean temperature, total rain and mean shortwave."""
    h = sim.history
    return [float(np.mean(h["temp_c"][-TICKS_PER_DAY:])),
            float(np.sum(h["rain_mm"][-TICKS_PER_DAY:])),
            float(np.mean(h["shortwave"][-TICKS_PER_DAY:]))]  # fmt: skip


def generate_run(seed: int, days: int, mode: str = "phase4") -> dict:
    scenario = scenario_for(seed)
    sim = new_sim(seed)
    rng = stream(seed, "dataset")
    xs, gs, plants, ticks, ws = [], [], [], [], []
    for day in range(days):
        advance_day(sim, rng, day, mode)
        snap = snapshot(sim)
        xs.append(snap.x)
        gs.append(snap.g)
        plants.append(sim.state.plant.c.astype(np.float32))
        ws.append(daily_weather(sim))
        ticks.append(sim.state.tick)
    applied = [i for i in sim.accepted if i.agent == "user"]
    out = {
        "x": np.array(xs), "g": np.array(gs), "plant": np.array(plants),
        "ticks": np.array(ticks), "seed": seed, "scenario": scenario,
        "grid": np.array([snap.rows, snap.cols]),
        "interventions": len(applied),
    }  # fmt: skip
    if mode != "phase4":  # Phase 4 files stay byte-for-byte what they were
        out["intents"] = json.dumps([i.model_dump(exclude={"rationale"}) for i in applied])
        out["w"] = np.array(ws)
    return out


def _job(args: tuple[int, int, str, str]) -> tuple[int, int, float]:
    seed, days, out, mode = args
    t0 = time.perf_counter()
    run = generate_run(seed, days, mode)
    path = Path(out) / f"run_{seed}.npz"
    np.savez_compressed(path, **{k: v for k, v in run.items()})
    return seed, days * TICKS_PER_DAY, time.perf_counter() - t0


def generate(seeds: list[int], days: int, out: Path = OUT, workers: int = 2,
             mode: str = "phase4") -> dict:  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    total = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for seed, ticks, secs in pool.map(_job, [(s, days, str(out), mode) for s in seeds]):
            total += ticks
            print(
                f"run {seed}: {ticks} ticks in {secs:.0f}s ({ticks / secs:.0f} ticks/s)", flush=True
            )
    wall = time.perf_counter() - t0
    stats = {"runs": len(seeds), "ticks": total, "wall_s": round(wall, 1),
             "ticks_per_s": round(total / wall), "workers": workers}  # fmt: skip
    print(json.dumps(stats), flush=True)
    return stats


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a research-mode dataset.")
    ap.add_argument("--seeds", default="0-27", help="e.g. 0-27 or 1000,1001")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mode", choices=list(MODES), default="phase4")
    ap.add_argument("--out", type=Path, default=None, help="default data/phase4 or data/phase5")
    args = ap.parse_args()
    out = args.out or (OUT if args.mode == "phase4" else OUT_FORCING)
    if "-" in args.seeds:
        lo, hi = (int(v) for v in args.seeds.split("-"))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(v) for v in args.seeds.split(",")]
    generate(seeds, args.days, out, args.workers, args.mode)


if __name__ == "__main__":
    main()
