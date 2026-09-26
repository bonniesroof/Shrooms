# Shrooms 🍄

A video-game-style ecosystem sandbox where plants, fungi, bacteria, insects and soil are agents on a shared world. Weather, temperature and contamination push on them.

Shrooms is a **learning sandbox first**. Each part of it teaches one of three skills:

- **Multi-agent simulation**: a deterministic, tick-based ecosystem. Interactions are local rules; the ecosystem-scale patterns emerge.
- **LLM agent orchestration**: a few LangGraph "keystone" agents (a narrator, a mycelial network) that reason on a slow clock and act through typed intents.
- **World models**: an ecosystem graph logged every tick, used to train a supervised GNN baseline and then a JEPA-inspired latent graph dynamics model.

## Status

Pre-alpha. Phase 0 (foundations). See [ROADMAP.md](ROADMAP.md).

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

```
sim/          deterministic core: world, agents, interactions, scenarios, engine
cognition/    LangGraph agents: narrator, keystone overlays, director
worldmodel/   graph builder, GNN baseline, JEPA, prediction heads, planners
server/       FastAPI + WebSocket bridge
client/       PixiJS game
notebooks/    one learning notebook per phase
tests/        unit, replay-determinism and mass-balance tests
data/         trajectory store (gitignored)
```

## Stack

Python 3.12 (NumPy, Numba, Pydantic, FastAPI, LangGraph, PyTorch Geometric) · TypeScript (Vite, PixiJS) · Parquet + DuckDB · uv, pnpm, Docker Compose, GitHub Actions.

## Getting started

*Planned. These commands will work once Phase 0 lands.*

```bash
uv sync                           # Python deps
uv run python -m sim.run --seed 42 --ticks 8760 --headless
uv run pytest                     # includes replay + mass-balance tests
cd client && pnpm install && pnpm dev
```

## Docs

- **Spec:** full design, including agents, environment, orchestration and world model (link TBD)
- **[ROADMAP.md](ROADMAP.md):** phased development plan

## Contributing

Keep commits small and scoped to one change. Keep this README current with what's in the repo.
