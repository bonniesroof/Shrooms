# Shrooms 🍄

A video-game-style ecosystem sandbox where plants, fungi, bacteria, insects and soil are agents on a shared world. Weather, temperature and contamination push on them.

Shrooms is a **learning sandbox first**. Each part of it teaches one of three skills:

- **Multi-agent simulation**: a deterministic, tick-based ecosystem. Interactions are local rules; the ecosystem-scale patterns emerge.
- **LLM agent orchestration**: a few LangGraph "keystone" agents (a narrator, a mycelial network) that reason on a slow clock and act through typed intents.
- **World models**: an ecosystem graph logged every tick, used to train a supervised GNN baseline and then a JEPA-inspired latent graph dynamics model.

## Status

**Phase 1 (living soil) complete.** A 64×64 world with layered hydrology, bacteria, saprotrophs, mycorrhizal fungi (model A market), insects, N and P cycles and a contamination field runs stably for years. Mass balance for C, N, P, water and contaminant is checked every tick, and every run replays bit-identically. A year takes about 25 s headless. Next up is Phase 2 (agent cognition). See [ROADMAP.md](ROADMAP.md).

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
  engine.py         tick loop: intents → weather → hydrology → guilds → balance check
  world.py          grid state: C/N/P pools, mineral N and P, contaminant, 3 water layers
  params.py         every tunable number, as frozen Pydantic models
  rng.py            one seeded PCG64 stream per system
  weather.py        seasonal/diurnal temperature, shortwave, Markov rain
  hydrology.py      soil layers, downslope runoff and subsurface flow, solute transport
  plants.py         nutrient-limited growth, root uptake, resorption, litterfall
  decomposers.py    bacteria, saprotrophs, SOM, N fixation, denitrification
  mycorrhiza.py     mycorrhizal fungi, model A biological market
  insects.py        herbivores with diapause and nutrient excretion
  contamination.py  toxicity and biodegradation
  transport.py      conservative lateral moves and shared-supply helpers
  env.py            per-tick context and environmental response curves
  ledger.py         mass-balance ledger (C, N, P, water, contaminant)
  intents.py        typed intents + validator (`disturb`, `spill`)
  replay.py         replay records and verification
  viewer.py         thin matplotlib debug viewer
  run.py            CLI
client/             pnpm + Vite + PixiJS stub (real views in Phase 3)
notebooks/          one learning notebook per phase
tests/              roadmap gates, component and per-guild ecology tests
data/               trajectory store (gitignored)
```

Planned: `cognition/` (LangGraph agents, Phase 2), `server/` (FastAPI + WebSocket, Phase 3), `worldmodel/` (graphs, GNN, JEPA, Phases 4–6).

## Stack

In use: Python 3.12 (NumPy, Pydantic, matplotlib) · TypeScript (Vite, PixiJS) · uv, pnpm, pre-commit, GitHub Actions.

Planned: Numba, FastAPI, LangGraph, PyTorch Geometric, Parquet + DuckDB, Docker Compose.

## Getting started

```bash
uv sync                                                        # Python deps
uv run python -m sim.run --seed 42 --ticks 8760 --headless     # 1-year run (~25 s) -> runs/seed42_t8760.json
uv run python -m sim.run --seed 42 --ticks 8760                # same, plus 12-panel debug-viewer PNGs in runs/
uv run python -m sim.run --replay runs/seed42_t8760.json       # verify bit-identical replay
uv run pytest                                                  # gate, replay and mass-balance tests
uv run pre-commit install                                      # ruff lint + format on commit
```

Submit intents from a JSON file with `--intents`, for example a clearing and a spill:

```json
[
  {"kind": "disturb", "tick": 2000, "x0": 16, "y0": 16, "x1": 48, "y1": 48, "fraction": 1.0},
  {"kind": "spill", "tick": 3000, "x": 40, "y": 20, "radius": 3, "mass_g": 500}
]
```

Records from runs with intents get an intent hash in their filename, so they never overwrite each other.

Notebooks: `uv sync --group notebooks && uv run jupyter lab`, then open `notebooks/phase1_living_soil.ipynb`. The Phase 0 notebook runs against tag `v0.0`.

Client stub: `cd client && pnpm install && pnpm dev`.

## Docs

- **Spec:** full design, including agents, environment, orchestration and world model (link TBD)
- **[ROADMAP.md](ROADMAP.md):** phased development plan
- **[Phase 0 notebook](notebooks/phase0_foundations.ipynb):** what was built, determinism, the carbon budget, and what broke
- **[Phase 1 notebook](notebooks/phase1_living_soil.ipynb):** nutrient cycling, the mycorrhizal market experiments, the brownfield signature, and the tuning log

## Contributing

Keep commits small and scoped to one change. Keep this README current with what's in the repo.
