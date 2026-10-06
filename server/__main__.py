"""Run the game server.

    uv run python -m server                       # brownfield, game mode, scripted agents
    uv run python -m server --mode research --seed 7
    uv run python -m server --model ollama:qwen2.5:7b-instruct
Then open http://localhost:8000 (after `cd client && pnpm build`), or run the
client dev server with `pnpm dev` and open http://localhost:5173.
"""

import argparse
import logging

import uvicorn

from cognition import scripted
from cognition.llm import Router, routes_from_spec
from server.app import ServerConfig, create_app


def main() -> None:
    ap = argparse.ArgumentParser(prog="server", description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenario", default="brownfield", help="scenario name, or 'none'")
    ap.add_argument("--mode", choices=["game", "research"], default="game")
    ap.add_argument("--model", default="scripted", help="scripted | ollama:<m> | openai:<m>")
    ap.add_argument("--base-url")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(message)s")
    router = Router(routes_from_spec(args.model, args.base_url), scripted=scripted.POLICIES)
    config = ServerConfig(seed=args.seed, mode=args.mode, router=router,
                          scenario=None if args.scenario == "none" else args.scenario)  # fmt: skip
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
