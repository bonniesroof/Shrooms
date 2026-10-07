# Shrooms architecture: built vs planned

Snapshot as of 2026-10-07 (commit `5d88c66`). Source of truth for plans is [ROADMAP.md](../ROADMAP.md); for repo contents, [README.md](../README.md).

## System diagram

Solid teal is built. Dashed gray is planned.

```mermaid
flowchart TB
    classDef built fill:#E1F5EE,stroke:#0F6E56,color:#04342C
    classDef planned fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A,stroke-dasharray: 4 3

    client["PixiJS client<br/>surface, underground, HUD"]:::built
    overlay["Forecast overlay<br/>Phase 4, from model heads"]:::planned

    subgraph srv["server/"]
        api["FastAPI + WebSocket<br/>session, tools, budget"]:::built
        director["Scenario Director<br/>game mode, brownfield"]:::built
    end

    subgraph cog["cognition/ (LangGraph, slow clock)"]
        narrator["Narrator<br/>fact-checked journal"]:::built
        keystone["Keystone agent<br/>mycelial network"]:::built
        router["Model router<br/>Ollama, vLLM, scripted"]:::built
    end

    subgraph sim["sim/ (deterministic, tick-based)"]
        engine["Engine<br/>tick loop, seeded RNG"]:::built
        env["Environment<br/>weather, hydrology"]:::built
        guilds["Life guilds<br/>plants, fungi, microbes, insects"]:::built
        validator["Validator<br/>rule-based intent gate"]:::built
        ledger["Mass-balance ledger<br/>C, N, P, water"]:::built
        replay["Replay and events<br/>bit-identical"]:::built
    end

    subgraph wm["worldmodel/ (planned, phases 4-6)"]
        dataset["Dataset generator<br/>Phase 4"]:::planned
        graph["Graph builder<br/>8x8 patches exist"]:::planned
        gnn["GNN, then JEPA<br/>Phases 4-5"]:::planned
        planner["CEM planner<br/>Phase 6"]:::planned
    end

    client <--> api
    api --> cog
    api --- director
    cog -->|typed intents| sim
    sim -.->|trajectories| wm
    wm -.->|forecasts| overlay
    overlay -.-> client
```

## Core design rules the diagram encodes

1. The sim never waits on an LLM. Agents submit typed intents; the validator accepts or rejects them.
2. Everything replays. A run is seed, sim version, parameters and accepted intents. Replay never calls an LLM.
3. Matter is conserved. CI fails on any unexplained gain or loss of C, N, P or water.
4. Two run modes. Research mode has no hidden interventions. Game mode adds the Scenario Director.

## Roadmap status

| Phase | Theme | Status | Notes |
|---|---|---|---|
| 0 | Foundations | Built | Grid, tick loop, seeded RNG, replay, debug viewer |
| 1 | Living soil | Built | Fungi (model A), N and P cycles, contamination, mass-balance tests |
| 2 | First agent brain | Built | Narrator and keystone agent have only run on the scripted stand-in so far |
| 3 | Game feel | Built, Gate B half-verified | 60 fps not yet checked on a real GPU (`/?bench`); brownfield playable end to end in CI |
| 4 | Graphs + baseline | Started | Patch IDs exist (`sim/patches.py`); no dataset generator, graph builder or GNN yet |
| 5 | JEPA | Planned | Static, temporal, then action-conditioned; Gate C vs GNN baseline |
| 6 | Closed loop | Planned | CEM planning, forecast heads into keystone `observe`, mycorrhizal models B and C |
| X | Optional | Planned | MARL, OpenClaw field station, more overlays, 128x128 world |

## Open items

- Verify Gate B's 60 fps half on a GPU machine.
- Run the Narrator and keystone agent against a real local model (Ollama or vLLM) and compare with the scripted policy.
- Start Phase 4: trajectory logging into `data/`, then the ecosystem graph builder.
