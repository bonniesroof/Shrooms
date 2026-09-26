# Shrooms 🍄

A video-game-style ecosystem sandbox where plants, fungi, bacteria, insects and soil are agents on a shared world. Weather, temperature and contamination push on them.

Shrooms is a **learning sandbox first**. Each part of it teaches one of three skills:

- **Multi-agent simulation**: a deterministic, tick-based ecosystem. Interactions are local rules; the ecosystem-scale patterns emerge.
- **LLM agent orchestration**: a few LangGraph "keystone" agents (a narrator, a mycelial network) that reason on a slow clock and act through typed intents.
- **World models**: an ecosystem graph logged every tick, used to train a supervised GNN baseline and then a JEPA-inspired latent graph dynamics model.

## Status

**Phase 0 (foundations) complete.** A 64×64 world runs a year in about 2.5 s headless. It has weather, soil water, and plants with explicit carbon pools. Every run replays bit-identically from its record, and carbon and water balances are checked every tick. Next up is Phase 1 (living soil). See [ROADMAP.md](ROADMAP.md).

## Core design rules

1. **The sim never waits on an LLM.** LLM agents submit typed intents such as `form_partnership` or `allocate_carbon`, and the sim validates them.
2. **Everything replays.** A run is its seed, sim version, parameters, and accepted intents with the tick each was applied. Replay never re-calls an LLM.
3. **Matter is conserved.** C, N, P and water are explicit pools and flows. CI fails on any unexplained gain or loss.
4. **There are two run modes.** *Research mode* has no hidden interventions and produces clean datasets. *Game mode* adds a Scenario Director, events and pacing.

## Architecture

```
PixiJS client  ⇄  FastAPI/WebSocket  ⇄  Sim core (tick loop + rule agents)
                                          ├─ Environment: weather, hydrology, contamination
                                          ├─ LangGraph cognition: narrator, keystone overlays
                                          └─ Trajectory store → GNN / JEPA world model → forecasts
```

## Repo layout

What exists today:

```
sim/                deterministic core
  engine.py         tick loop: intents → weather → water → carbon → balance check
  world.py          grid state (per-cell float64 arrays) and world hashing
  params.py         every tunable number, as frozen Pydantic models
  rng.py            one seeded PCG64 stream per system
  weather.py        seasonal/diurnal temperature, shortwave, Markov rain
  water.py          soil water bucket per cell
  plants.py         plant and soil carbon pools and flows, dispersal
  ledger.py         mass-balance ledger (C and water)
  intents.py        typed intents + validator (Phase 0: `disturb`)
  replay.py         replay records and verification
  viewer.py         thin matplotlib debug viewer
  run.py            CLI
client/             pnpm + Vite + PixiJS stub (real views in Phase 3)
notebooks/          phase0_foundations.ipynb
tests/              Phase 0 gate, replay-determinism and mass-balance tests
data/               trajectory store (gitignored)
```

Planned: `cognition/` (LangGraph agents, Phase 2), `server/` (FastAPI + WebSocket, Phase 3), `worldmodel/` (graphs, GNN, JEPA, Phases 4–6).

## Stack

In use: Python 3.12 (NumPy, Pydantic, matplotlib) · TypeScript (Vite, PixiJS) · uv, pnpm, pre-commit, GitHub Actions.

Planned: Numba, FastAPI, LangGraph, PyTorch Geometric, Parquet + DuckDB, Docker Compose.

## Getting started

```bash
uv sync                                                        # Python deps
uv run python -m sim.run --seed 42 --ticks 8760 --headless     # 1-year run -> runs/seed42_t8760.json
uv run python -m sim.run --seed 42 --ticks 8760                # same, plus debug-viewer PNGs in runs/
uv run python -m sim.run --replay runs/seed42_t8760.json       # verify bit-identical replay
uv run pytest                                                  # gate, replay and mass-balance tests
uv run pre-commit install                                      # ruff lint + format on commit
```

Submit intents from a JSON file with `--intents`, for example a clearing at tick 2000:

```json
[{"kind": "disturb", "tick": 2000, "x0": 16, "y0": 16, "x1": 48, "y1": 48, "fraction": 1.0}]
```

Notebook: `uv sync --group notebooks && uv run jupyter lab`, then open `notebooks/phase0_foundations.ipynb`.

Client stub: `cd client && pnpm install && pnpm dev`.

## Docs

- **Spec:** full design, including agents, environment, orchestration and world model (link TBD)
- **[ROADMAP.md](ROADMAP.md):** phased development plan
- **[Phase 0 notebook](notebooks/phase0_foundations.ipynb):** what was built, determinism, the carbon budget, and what broke

## Contributing

Keep commits small and scoped to one change. Keep this README current with what's in the repo.
