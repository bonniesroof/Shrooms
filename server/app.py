"""FastAPI app: one live game session, streamed to browsers over a WebSocket.

    GET  /api/health       liveness and sim version
    GET  /api/record       the session's replay record (JSON)
    WS   /ws               frames out (~10 per second), commands in
    GET  /                 the built PixiJS client (client/dist), if present

Commands (JSON): {"cmd": "speed", "value": 24} | {"cmd": "step", "days": 1}
| {"cmd": "tool", "tool": "compost", "x": 30, "y": 22} | {"cmd": "whisper", "text": "..."}
| {"cmd": "mode", "value": "research"} | {"cmd": "new_game", "seed": 7, "mode": "game"}
Every command gets a {"type": "ack"} or {"type": "error"} reply.
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from cognition.llm import Router
from server.session import SPEEDS, TOOLS, GameSession
from sim import SIM_VERSION, TICKS_PER_DAY

log = logging.getLogger("shrooms.server")
FRAME_INTERVAL = 0.1  # seconds
CLIENT_DIST = Path(__file__).resolve().parent.parent / "client" / "dist"


@dataclass
class ServerConfig:
    seed: int = 42
    scenario: str | None = "brownfield"
    mode: str = "game"
    router: Router | None = None
    start_speed: int = 0  # paused until the player presses play


class Hub:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.session = self._new(config.seed, config.mode)
        self.speed = config.start_speed
        self.clients: set[WebSocket] = set()
        self.carry = 0.0
        self.task: asyncio.Task | None = None

    def _new(self, seed: int, mode: str) -> GameSession:
        return GameSession(seed=seed, scenario=self.config.scenario, mode=mode,
                           router=self.config.router)  # fmt: skip

    async def run(self) -> None:
        while True:
            await asyncio.sleep(FRAME_INTERVAL)
            try:
                if self.speed:
                    self.carry += self.speed * FRAME_INTERVAL
                    n, self.carry = int(self.carry), self.carry - int(self.carry)
                    if n:
                        await asyncio.to_thread(self.session.advance, n)
                    status = self.session.status
                    if status and status["status"] != "playing":
                        self.speed = 0  # stop the clock when the game ends
                if self.clients:
                    await self.broadcast(await asyncio.to_thread(self.frame))
            except Exception:  # keep serving; the error is in the log
                log.exception("sim loop error")

    def frame(self) -> dict:
        f = self.session.frame()
        f["speed"] = self.speed
        return f

    async def broadcast(self, message: dict) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                self.clients.discard(ws)

    def hello(self) -> dict:
        return {
            "type": "hello",
            "sim_version": SIM_VERSION,
            "speeds": SPEEDS,
            "tools": {k: {"label": t.label, "cost": t.cost, "hint": t.hint,
                          "research_only": t.research_only} for k, t in TOOLS.items()},
        }  # fmt: skip

    async def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        s = self.session
        if cmd == "speed":
            value = int(msg.get("value", 0))
            if value not in SPEEDS:
                return {"type": "error", "error": f"speed must be one of {SPEEDS}"}
            self.speed = value
        elif cmd == "step":
            days = max(1, min(30, int(msg.get("days", 1))))
            await asyncio.to_thread(s.advance, days * TICKS_PER_DAY)
        elif cmd == "tool":
            result = s.use_tool(str(msg.get("tool")), float(msg["x"]), float(msg["y"]))
            if not result["ok"]:
                return {"type": "error", "cmd": cmd, "error": result["error"]}
        elif cmd == "whisper":
            text = str(msg.get("text", "")).strip()
            if not text:
                return {"type": "error", "cmd": cmd, "error": "empty whisper"}
            s.whisper(text)
        elif cmd == "mode":
            try:
                s.set_mode(str(msg.get("value")))
            except ValueError:
                return {"type": "error", "cmd": cmd, "error": "mode must be game or research"}
        elif cmd == "new_game":
            mode = str(msg.get("mode", s.mode))
            s.close()
            self.session = self._new(int(msg.get("seed", self.config.seed)), mode)
            self.speed = 0
        else:
            return {"type": "error", "error": f"unknown command {cmd!r}"}
        return {"type": "ack", "cmd": cmd}


def create_app(config: ServerConfig | None = None) -> FastAPI:
    hub = Hub(config or ServerConfig())

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.task = asyncio.create_task(hub.run())
        yield
        hub.task.cancel()
        hub.session.close()

    app = FastAPI(title="Shrooms", lifespan=lifespan)
    app.state.hub = hub

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "sim_version": SIM_VERSION, "tick": hub.session.sim.state.tick}

    @app.get("/api/record")
    def record() -> JSONResponse:
        return JSONResponse(hub.session.record().model_dump(mode="json"))

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json(hub.hello())
        await websocket.send_json(await asyncio.to_thread(hub.frame))
        hub.clients.add(websocket)
        try:
            while True:
                msg = await websocket.receive_json()
                try:
                    reply = await hub.handle(msg)
                except (KeyError, TypeError, ValueError) as err:
                    reply = {"type": "error", "error": f"bad command: {err}"}
                await websocket.send_json(reply)
                if reply["type"] == "ack" and msg.get("cmd") != "speed":
                    await websocket.send_json(await asyncio.to_thread(hub.frame))
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(websocket)

    if CLIENT_DIST.exists():
        app.mount("/", StaticFiles(directory=CLIENT_DIST, html=True), name="client")
    return app
