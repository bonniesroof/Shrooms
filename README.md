# Shrooms 🍄

A video-game-style ecosystem sandbox where plants, fungi, bacteria, insects and soil are agents on a shared world. Weather, temperature and contamination push on them.

Shrooms is a **learning sandbox first**. Each part of it teaches one of three skills:

- **Multi-agent simulation**: a deterministic, tick-based ecosystem. Interactions are local rules; the ecosystem-scale patterns emerge.
- **LLM agent orchestration**: a few LangGraph "keystone" agents (a narrator, a mycelial network) that reason on a slow clock and act through typed intents.
- **World models**: an ecosystem graph logged every tick, used to train a supervised GNN baseline and then a JEPA-inspired latent graph dynamics model.

## Status

**Phase 4 (graphs + baseline) done.** The ecosystem is now a typed graph: 64 patch nodes, an atmosphere node, and adjacent, downslope and hyphal edges. A research-mode generator produces batched headless runs, and a supervised GNN forecasts biomass, contamination, moisture, die-back and fungal links at 24 h, 7 d and 30 d. On four test worlds it never saw, it **beats persistence on every head at 7 and 30 days**: errors are 36% smaller at 7 days and 43% smaller at 30 days, averaged over heads. Forecasts run live in the game as a map overlay. The one loss is fungal links at 24 h, where persistence is near-perfect. The 1,000 ticks/s generator target is **not met on this 2-core sandbox** (557 ticks/s across 2 workers; it scales with cores).

**Phase 5 (JEPA) in progress: steps 1–3 of 4 done.** A static graph JEPA learns patch latents without labels. It hides a block of patches in a daily snapshot and predicts their latents from the rest; an EMA target encoder supplies the latents to predict. Linear probes on test worlds show where it stands. For a **hidden** patch, its predicted latent gives mean R² 0.76 on current biomass, contamination, moisture and network membership, against 0.55 for the raw features of its visible 1-hop neighbours and 0.38 for the same network untrained. Moisture is the exception: it is spatially smooth, so raw neighbours do as well as JEPA (R² 0.825 vs 0.826), and beat it in a 15%-mask run (0.839 vs 0.828). For a **visible** patch, the latents keep the state (mean R² 0.89), but no better than a random-init network (0.92) or the raw features themselves (0.90). No collapse: the effective rank is 37 of 64.

Step 2, the temporal JEPA, predicts the latents of the snapshot at t + 1, 7 and 30 days from the latents at t. Probed linearly onto the Phase 4 forecast heads, it **does not yet beat the supervised GNN or linear probes on raw features**. Mean skill against persistence on the test worlds at 1, 7 and 30 days: JEPA probe 0.22 / 0.21 / 0.23, raw-feature probe 0.28 / 0.27 / 0.30, GNN 0.26 / 0.36 / 0.43. One caveat inflates all of these. Die-back is zero almost everywhere, so under MAE an intercept-only "no die-back" probe already scores about 0.5 skill on it, and every probe gets that for free; a constant probe's mean is 0.13 / 0.10 / 0.10. Without die-back the means are: JEPA 0.15 / 0.14 / 0.17, raw 0.22 / 0.21 / 0.25, raw + JEPA 0.23 / 0.23 / 0.27, GNN 0.22 / 0.32 / 0.42. The latents add a little on top of raw features: on contamination about equally at every horizon, and on links mostly at 30 days. Metrics come from the full-precision model; the checkpoint is stored in fp16. 

Step 3, the action-conditioned JEPA, also tells the predictor what happens between t and t + k: every intervention in the window, per patch and kind, and the window's mean weather. It is trained on a new forcing dataset with an intervention every 10 days at probability 0.6 and the applied intents recorded (`dataset.py --mode forcing`: 28 two-year worlds, 11 min). It is tested on 52 spill-vs-no-spill counterfactual pairs, branched from held-out worlds and rolled forward by the deterministic sim. Results on the test worlds:

- **Counterfactuals.** Read through a least-squares probe of the contamination level, the predicted spill effect has the right sign on 99.6% of affected patches and a spatial correlation of 0.88 with the true effect map. It finds the most affected patch 62% of the time, but overshoots the magnitude by 1.3–1.4×; effect skill is 0.25 at 7 days and 0.22 at 30 days. The step-2 model can't tell the branches apart, so it scores 0 by construction. The Phase 4 forecast probes barely react to the spill: they are LAD fits of mostly-zero changes, so a rare event moves them by almost nothing. The model doesn't get the spill's small effect on biomass (true mean 0.002 log units at 30 days); its predictions there are noise.
- **Forecast skill.** Being told the forcing helps the JEPA latents only marginally: mean skill 0.21 / 0.20 / 0.22 at 1 / 7 / 30 days, against 0.21 / 0.20 / 0.22 for step 2 on the same data; without die-back, 0.14 / 0.13 / 0.16 against 0.13 / 0.12 / 0.16. A linear probe on raw inputs plus the same forcing features does far better: 0.41 / 0.42 / 0.38 (0.39 / 0.40 / 0.36 without die-back). That is mostly because the window's weather predicts moisture almost directly (skill 0.91 at 1 day), and it beats the GNN, which isn't told the future weather. The JEPA's latents don't pass that weather through to a linear readout.

Next comes collapse monitoring and the comparison against the GNN and a generative baseline.

Still open from Phase 3: the 60 fps half of Gate B is unverified on a real GPU (open `/?bench` on a machine with one). Phase 2's agents have only run on the scripted stand-in. See [ROADMAP.md](ROADMAP.md).

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
worldmodel/         graphs, dataset, forecasters
  graph.py          typed patch/atmosphere nodes; adjacent, downslope, hyphal edges
  dataset.py        research-mode batched runs with random interventions, daily snapshots;
                    `--mode forcing` records the applied intents and daily weather (data/phase5)
  targets.py        forecast heads, persistence baselines, skill scores
  gnn.py            supervised GNN (PyTorch): typed message passing, node and link heads
  forecast.py       numpy twin of the GNN for torch-free inference (server, CI)
  train.py          training, model selection, export
  forecaster.npz    the trained model (+ forecaster_metrics.json)
  jepa.py           graph JEPAs (PyTorch): static (masked patches), temporal (t -> t + k),
                    and action-conditioned (given interventions and weather)
  probes.py         linear probes (state and forecast heads) and collapse stats, numpy
  train_jepa.py     self-supervised training on snapshots, probes vs random init and raw features
  jepa_static.pt    the trained JEPA (+ jepa_static_metrics.json)
  train_jepa_temporal.py  temporal JEPA (latents at t -> t + k), forecast probes vs persistence and GNN
  jepa_temporal.pt  the trained temporal JEPA (+ jepa_temporal_metrics.json)
  forcing.py        interventions (per patch, per kind) and weather between two snapshots
  counterfactual.py spill vs no-spill branches from the same moment, rolled out by the sim
  train_jepa_action.py  action-conditioned JEPA; forecast probes and counterfactual tests
  jepa_action.pt    the trained action-conditioned JEPA (+ jepa_action_metrics.json)
tests/              roadmap gates, component and per-guild ecology tests
data/               trajectory store (gitignored)
```

Planned: collapse monitoring and the comparison against the GNN and a generative baseline (rest of Phase 5), in `worldmodel/`.

## Stack

In use: Python 3.12 (NumPy, Pydantic, matplotlib, LangGraph + SQLite checkpointer, httpx, FastAPI, uvicorn, Numba; PyTorch for training only) · Ollama / vLLM · TypeScript (Vite, PixiJS v8) · uv, pnpm, pre-commit, GitHub Actions.

Planned: Parquet + DuckDB, Docker Compose. PyTorch Geometric turned out unnecessary at 64 nodes: dense per-type adjacency in plain PyTorch is simpler.

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

**Forecasts.** Under *Forecast (GNN)*, pick a horizon (24 h, 7 d, 30 d) and a head. Patches are tinted green for good news and red for bad: more plants or water, less contamination. Die-back risk shows in red; fungal links show as forming (green) or breaking (red). Hover a patch for its number. The note gives that head's tested skill against "no change", so you know how far to trust it.

### World model

```bash
uv sync --group worldmodel                                  # adds PyTorch (training only)
uv run python -m worldmodel.dataset --seeds 0-27 --workers 2  # 28 two-year worlds, ~15 min on 2 cores
uv run python -m worldmodel.train                             # ~10 min on CPU; writes worldmodel/forecaster.npz
uv run python -m worldmodel.train_jepa                        # ~9 min on CPU; writes worldmodel/jepa_static.pt
uv run python -m worldmodel.train_jepa_temporal               # ~5 min on CPU; writes worldmodel/jepa_temporal.pt
uv run python -m worldmodel.dataset --mode forcing --seeds 0-27   # ~11 min; worlds with recorded interventions
uv run python -m worldmodel.counterfactual                    # ~3 min; spill vs no-spill pairs on test worlds
uv run python -m worldmodel.train_jepa_action                 # ~5 min on CPU; writes worldmodel/jepa_action.pt
```

Splits are by world: seeds 0–19 train, 20–23 select the best epoch, 24–27 are reported. The game server and CI use the exported numpy model, so they don't need PyTorch.

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
