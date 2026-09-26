"""Run the sim with its agents.

    uv run python -m cognition.run --model scripted --ticks 8760
    uv run python -m cognition.run --model ollama:qwen2.5:7b-instruct --log-level DEBUG
    uv run python -m cognition.run --model openai:Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000

Every model route falls back to the scripted policy if the server is unreachable
or keeps returning unusable output. The fallback is logged, never silent.
"""

import argparse
import logging
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from cognition import scripted
from cognition.llm import Router, routes_from_spec
from cognition.runner import AgentConfig, run_with_agents
from sim import TICKS_PER_YEAR
from sim.replay import ReplayMismatch, replay


def setup_logging(level: str, log_file: Path | None) -> None:
    fmt = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", handlers=handlers, force=True)
    for noisy in ("httpx", "httpcore", "langgraph", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cognition.run", description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=TICKS_PER_YEAR)
    ap.add_argument(
        "--model",
        default="ollama:qwen2.5:7b-instruct",
        help="scripted | ollama:<model> | openai:<model>",
    )
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--no-narrator", action="store_true")
    ap.add_argument("--no-mycelium", action="store_true")
    ap.add_argument(
        "--verify-replay",
        action="store_true",
        help="replay the record afterwards (no model calls) and compare hashes",
    )
    args = ap.parse_args(argv)

    stem = f"agents_seed{args.seed}_t{args.ticks}_{args.model.replace(':', '-').replace('/', '-')}"
    setup_logging(args.log_level, args.out / f"{stem}.log")
    log = logging.getLogger("shrooms.cli")
    router = Router(routes_from_spec(args.model, args.base_url), scripted=scripted.POLICIES)
    cfg = AgentConfig(enable_narrator=not args.no_narrator, enable_mycelium=not args.no_mycelium)

    start = time.perf_counter()
    art = run_with_agents(args.seed, args.ticks, router, args.out, stem, config=cfg)
    elapsed = time.perf_counter() - start
    art.close()
    sim = art.sim

    agent_intents = [i for i in sim.accepted if i.agent == "mycelium"]
    proposed = sum(
        len(t["accepted"]) + len(t["rejected"])
        for t in (art.mycelium.traces if art.mycelium else [])
    )
    graph_rejects = Counter(
        v.split(":")[0]
        for t in (art.mycelium.traces if art.mycelium else [])
        for r in t["rejected"]
        for v in r["violations"]
    )
    calls = router.calls
    print(f"\n=== run {stem} ({elapsed:.1f}s) ===")
    if art.mycelium:
        print(
            f"mycelium: {len(art.mycelium.traces)} decisions, {proposed} proposals, "
            f"{len(agent_intents)} applied, {len(sim.rejected)} refused at apply time"
        )
        print("  applied by kind:", dict(Counter(i.kind for i in agent_intents)))
        print("  rejected in-graph by rule:", dict(graph_rejects) or "none")
    if art.narrator:
        grounded = sum(e["grounded"] for e in art.narrator.entries)
        print(
            f"narrator: {len(art.narrator.entries)} entries, {grounded} passed the fact check "
            "first time or on retry"
        )
    if calls:
        fell = sum(c["fell_back"] for c in calls)
        lat = [c["latency_s"] for c in calls]
        print(
            f"llm: {len(calls)} calls via {dict(Counter(c['route'] for c in calls))}, "
            f"{fell} fallbacks, median latency {statistics.median(lat) * 1000:.1f} ms"
        )
    print(f"events: {dict(Counter(e.kind for e in sim.events.events))}")
    for key, path in art.paths.items():
        print(f"  {key:>11}: {path}")
    print(f"  {'log':>11}: {args.out / (stem + '.log')}")

    if args.verify_replay:
        t0 = time.perf_counter()
        try:
            replay(art.record)
        except ReplayMismatch as err:
            log.error("REPLAY FAILED: %s", err)
            return 1
        print(
            f"replay: OK without any model calls ({time.perf_counter() - t0:.1f}s, "
            f"{len(art.record.intents)} recorded intents)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
