# Shrooms 🍄

A video-game-style ecosystem sandbox where plants, fungi, bacteria, insects and soil are agents on a shared world. Weather, temperature and contamination push on them.

Shrooms is a **learning sandbox first**. Each part of it teaches one of three skills:

- **Multi-agent simulation**: a deterministic, tick-based ecosystem. Interactions are local rules; the ecosystem-scale patterns emerge.
- **LLM agent orchestration**: a few LangGraph "keystone" agents (a narrator, a mycelial network) that reason on a slow clock and act through typed intents.
- **World models**: an ecosystem graph logged every tick, used to train a supervised GNN baseline and then a JEPA-inspired latent graph dynamics model.

## Status

**Phase 3 (game feel) built; Gate B half-verified.** A FastAPI/WebSocket server streams the live sim to a PixiJS client with a surface view and an underground view (the glowing hyphal network, with the keystone agent's nutrient shuttles as travelling pulses). There's a HUD with time controls, remediation tools, an inspector showing the agent's observation → intent → rationale, a whisper box to the agent, and a research/game mode switch. A Scenario Director paces game mode. The first scenario, **brownfield remediation**, is playable end to end: in CI, a scripted gardener wins it and an idle one loses.

The 60 fps half of Gate B is **not yet verified on a real GPU**. Measured so far in headless Chromium on SwiftShader (a CPU software rasterizer), the dense benchmark reaches 35–43 fps at 800×450 with 4,700–9,700 entities; per-frame JS work is about 0.1 ms. At 1600×900 software fill rate drops it to single digits. Open `/?bench` on a machine with a GPU to check. Phase 2's agents have only run on the scripted stand-in so far. See [ROADMAP.md](ROADMAP.md).

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
  intents.py        typed intents: `disturb`, `spill`; the network's `shuttle_nutrients`,
                    `relocate_hyphae`, `set_trade_bias`; player tools `inoculate`, `amend`,
                    `seed`, `irrigate`, `excavate`; director events `pest_outbreak`, `downpour`
  replay.py         replay records and verification
  viewer.py         thin matplotlib debug viewer
  run.py            CLI
  patches.py        8x8 patches with stable IDs ("r3c5")
  validator.py      rule-based intent validator (authority, geometry, network, resources, bounds, rate)
  events.py         event log with daily detection (storms, droughts, market, cleanup...)
cognition/          LLM agents (LangGraph)
  llm.py            model router: Ollama, OpenAI-compatible (vLLM), scripted fallback
  keystone.py       mycelial-network keystone agent
  narrator.py       Narrator with a fact-checked field journal
  observe.py        compact observations for agents
  prompts.py        prompt text
  scripted.py       deterministic stand-in policies
  runner.py         runs sim + agents on slow clocks, SQLite checkpoints
  run.py            CLI
server/             game server
  app.py            FastAPI + WebSocket: frames out, commands in; serves the built client
  session.py        live session: sim, agents off the sim thread, director, budget, tools
  scenarios.py      brownfield remediation: setup intents, objectives, clock
  director.py       Scenario Director (game mode only): pacing through validated events
  players.py        scripted players used to balance and test scenarios
client/             PixiJS game (TypeScript, Vite)
  src/world.ts      surface and underground views, hyphal network, entities
  src/hud.ts        time controls, tools, objectives, inspector, whisper, feed
  src/net.ts        WebSocket connection and frame decoding
notebooks/          one learning notebook per phase
tests/              roadmap gates, component and per-guild ecology tests
data/               trajectory store (gitignored)
```

Planned: `worldmodel/` (graphs, GNN, JEPA, Phases 4–6).

## Stack

In use: Python 3.12 (NumPy, Pydantic, matplotlib, LangGraph + SQLite checkpointer, httpx, FastAPI, uvicorn) · Ollama / vLLM · TypeScript (Vite, PixiJS v8) · uv, pnpm, pre-commit, GitHub Actions.

Planned: Numba, PyTorch Geometric, Parquet + DuckDB, Docker Compose.

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

Notebooks: `uv sync --group notebooks && uv run jupyter lab`, then open `notebooks/phase2_agent_brain.ipynb`. The Phase 0 and Phase 1 notebooks run against tags `v0.0` and `v0.1`.

Agents (a year makes ~117 model calls; both agents share one model so a 10 GB GPU never swaps):

```bash
uv run python -m cognition.run --model scripted --verify-replay                    # no GPU needed
ollama pull qwen2.5:7b-instruct
uv run python -m cognition.run --model ollama:qwen2.5:7b-instruct --ticks 2016 --log-level DEBUG
uv run python -m cognition.run --model openai:<model> --base-url http://localhost:8000   # vLLM
```

Each run writes a replay record, the field journal, a per-decision trace (JSONL), LangGraph checkpoints (SQLite) and a full log to `runs/`. If the model server is unreachable, the run logs a warning and continues on the scripted policy.

### Play

```bash
cd client && pnpm install && pnpm build && cd ..
uv run python -m server                     # brownfield, game mode, scripted agents
# open http://localhost:8000
uv run python -m server --model ollama:qwen2.5:7b-instruct   # a real model for the agents
uv run python -m server --mode research --seed 7             # no director; tools are free
```

For client development, run `pnpm dev` in `client/` alongside the server and open http://localhost:5173; it proxies `/ws` and `/api` to :8000.

**How to play brownfield.** An old industrial pad (the gold square) has been stripped to subsoil and soaked in hydrocarbons. Within three years, get the contaminant below 25%, plant cover above 60%, and 10 of the 16 site patches onto the fungal network. Pick a tool on the left and click the map. Bacteria and compost drive cleanup, but fungi only establish on clean ground, so the objectives have to come in order. Space pauses and plays, Tab flips between surface and underground, Esc drops the tool. Watch the inspector to see what the mycelial network decides each week, and whisper to it, for example "help r3c4". It may refuse, and it will say why. A budget grant arrives monthly; the director answers if you race ahead or fall behind.

`/?bench` renders a dense synthetic world without a server and reports fps (`&view=underground`, `&aa=0`, `&regen=0` vary it). `/api/record` downloads the session's replay record.

## Docs

- **Spec:** full design, including agents, environment, orchestration and world model (link TBD)
- **[ROADMAP.md](ROADMAP.md):** phased development plan
- **[Architecture](docs/architecture.md):** system diagram of what's built and what's planned, plus phase status
- **[Phase 0 notebook](notebooks/phase0_foundations.ipynb):** what was built, determinism, the carbon budget, and what broke
- **[Phase 1 notebook](notebooks/phase1_living_soil.ipynb):** nutrient cycling, the mycorrhizal market experiments, the brownfield signature, and the tuning log
- **[Phase 2 notebook](notebooks/phase2_agent_brain.ipynb):** a decision end to end, validator feedback, replay without a model, whether the agent helps, and what broke

## Contributing

Keep commits small and scoped to one change. Keep this README current with what's in the repo.
