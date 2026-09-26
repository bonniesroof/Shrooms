# Shrooms Roadmap

**Goal: learning sandbox first.** Every phase ends with something runnable, teaches one clear skill, and ships a notebook in `notebooks/` covering what was built and what broke. The sim, the agents, the UI and the world model all show up early in crude form and get refined later, rather than one being perfected before the next starts.

Pace assumes side-project time: roughly 22 weeks for the core, plus an optional Phase X.

| Phase | Weeks | Theme | You learn |
|---|---|---|---|
| 0 | 1–2 | Foundations | ABM basics, determinism |
| 1 | 3–5 | Living soil | emergence, tuning, conservation |
| 2 | 6–7 | First agent brain | LangGraph fundamentals |
| 3 | 8–10 | Game feel | real-time streaming, game UX |
| 4 | 11–13 | Graphs + baseline | GNNs, datasets, forecasting |
| 5 | 14–18 | JEPA | self-supervised world models |
| 6 | 19–22 | Closed loop | model-based planning |
| X | — | Optional | MARL, agent ops, scaling |

---

## Phase 0: Foundations (weeks 1–2)

**Build**
- Repo scaffolding: uv, pnpm, CI, pre-commit.
- A 64×64 grid, the tick loop (1 tick = 1 sim-hour), and seeded RNG per system.
- Weather generator, plus plants and water with carbon accounting.
- Replay record: seed, sim version, parameters and intents.
- A **thin debug viewer**: colored cells in PixiJS or matplotlib. It needs to be ugly and useful, nothing more.

**Done when:** a headless 1-year run completes, replays bit-identically, and the carbon mass-balance test passes in CI.

## Phase 1: Living soil (weeks 3–5)

**Build**
- Soil column layers and hydrology.
- Mycorrhizal fungi, starting with model A (biological market).
- Saprotrophs, bacteria pools and insects.
- Decomposition, the nitrogen and phosphorus cycles, and a contamination field.
- Mass-balance tests for C, N, P and water.

**Gate A:** a stable 1-year run with no default collapse, mass balance passing, and replay identical.

## Phase 2: First agent brain (weeks 6–7)

**Build**
- A typed intent schema and a rule-based validator.
- A **Narrator** as a LangGraph graph that reads the event log and writes the field journal.
- One **keystone overlay**, the mycelial network, following observe → deliberate → propose → validate → commit, on a slow clock.
- A SQLite checkpointer, and local model routing through Ollama or vLLM.

**Done when:** the keystone agent makes validated decisions that change the sim, and replay reproduces them without calling the LLM.

## Phase 3: Game feel (weeks 8–10)

**Build**
- PixiJS surface view and underground view with the glowing hyphal network.
- HUD, time controls, and an inspector that shows observation → intent → generated rationale.
- God tools, plus an HITL "whisper" to the keystone agent.
- A research/game mode switch, and a basic Scenario Director in game mode only.
- Brownfield remediation as the first scenario.

**Gate B:** 60 fps rendering with about 2,000 entities, and the brownfield scenario is playable end to end.

## Phase 4: Graphs + baseline (weeks 11–13)

**Build**
- An ecosystem graph builder: typed nodes and edges, and fixed 8×8 m patches with stable IDs.
- A research-mode dataset generator using batched headless runs, with a target of at least 1,000 ticks/s.
- A **supervised GNN** forecaster with prediction heads (biomass, mortality, contamination, moisture, fungal links) at 24 h, 7 d and 30 d.
- A crude forecast overlay in the UI, driven by the heads.

**Done when:** the GNN beats persistence at 7 and 30 days, and forecasts are visible in-game.

## Phase 5: JEPA (weeks 14–18)

**Build**, in this order:
- Static JEPA on snapshots, evaluated with linear probes.
- A temporal version that predicts latents at t+k.
- An action-conditioned version that adds weather and interventions, tested on counterfactuals (spill vs no spill).
- Collapse monitoring (embedding variance and effective rank) and a comparison against the GNN baseline and a generative baseline.

**Gate C:** the JEPA model with prediction heads beats the GNN baseline at 7 and 30 days. If it doesn't, write up why; that counts as a result.

## Phase 6: Closed loop (weeks 19–22)

**Build**
- Latent-space planning (CEM) on the brownfield objective: minimize contamination, maintain biodiversity, and minimize intervention cost.
- Forecast heads fed into the keystone agent's `observe` step, measuring whether its decisions improve.
- Mycorrhizal models B (source–sink) and C (hybrid), compared with A on identical seeds.
- A drought-survival scenario.

**Done when:** the planner finds interventions that beat hand-picked ones, and the A/B/C comparison is written up.

## Phase X: Optional

- MARL with PettingZoo and PPO for tree and fungal policies.
- An OpenClaw "field station" that runs overnight sweeps and messages a digest, sandboxed with no repo tokens.
- More cognitive overlays, such as an old oak.
- A 128×128 world and multi-year scenarios.

---

## Working rules

- Each phase ends with a notebook, an updated README, and a tagged release (`v0.<phase>`).
- Gates are hard. Don't start UI polish before Gate A, or planning before Gate C.
- Keep commits small, and push to `main`.
