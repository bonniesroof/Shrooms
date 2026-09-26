"""Command-line entry point.

uv run python -m sim.run --seed 42 --ticks 8760 --headless
uv run python -m sim.run --seed 42 --ticks 8760            # also writes PNGs
uv run python -m sim.run --replay runs/seed42.json
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from pydantic import TypeAdapter

from sim import SIM_VERSION, TICKS_PER_YEAR
from sim.intents import Intent
from sim.replay import ReplayMismatch, ReplayRecord, record_run, replay
from sim.world import POOLS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sim.run", description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=TICKS_PER_YEAR)
    ap.add_argument("--headless", action="store_true", help="skip the debug viewer PNGs")
    ap.add_argument("--intents", type=Path, help="JSON list of intents to submit")
    ap.add_argument("--out", type=Path, default=Path("runs"), help="output directory")
    ap.add_argument("--replay", type=Path, help="replay a record and verify its hashes")
    args = ap.parse_args(argv)

    start = time.perf_counter()
    if args.replay:
        record = ReplayRecord.load(args.replay)
        try:
            sim = replay(record)
        except ReplayMismatch as err:
            print(f"REPLAY FAILED: {err}", file=sys.stderr)
            return 1
        print(f"replay OK: {record.ticks} ticks, final hash {record.final_hash[:16]}")
    else:
        intents = []
        if args.intents:
            raw = json.loads(args.intents.read_text())
            intents = TypeAdapter(list[Intent]).validate_python(raw)
        sim, record = record_run(args.seed, args.ticks, intents=intents)
        stem = f"seed{args.seed}_t{args.ticks}"
        if record.intents:  # different intents must not overwrite each other's records
            blob = json.dumps([i.model_dump() for i in record.intents], sort_keys=True)
            stem += "_i" + hashlib.sha256(blob.encode()).hexdigest()[:8]
        path = args.out / f"{stem}.json"
        record.save(path)
        print(f"sim {SIM_VERSION} seed {args.seed}: {args.ticks} ticks -> {path}")
        for intent, reason in sim.rejected:
            print(f"  rejected intent at tick {intent.tick}: {reason}")

    elapsed = time.perf_counter() - start
    s = sim.state
    print(
        f"  {elapsed:.1f}s ({s.tick / max(elapsed, 1e-9):.0f} ticks/s)  "
        "mass balance OK (C, N, P, water, contaminant)"
    )
    print("  mean g C/cell: " + "  ".join(f"{name} {s.pool(name).c.mean():.1f}" for name in POOLS))
    print(
        f"  mineral N {s.mineral_n.mean():.3f}  mineral P {s.mineral_p.mean():.4f} g/cell  "
        f"contaminant {s.contaminant.sum():.0f} g  moisture {s.moisture().mean():.2f}"
    )

    if not args.headless:
        from sim.viewer import save_views

        stem = args.replay.stem if args.replay else path.stem
        for p in save_views(sim, args.out, stem=stem):
            print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
