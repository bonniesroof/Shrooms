"""Model router: JSON repair, retries, fallback, and both HTTP backends (fake servers)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from cognition.llm import LLMError, ModelRoute, Router, extract_json, routes_from_spec


@pytest.mark.parametrize(
    "text",
    [
        '{"intents": []}',
        '```json\n{"intents": []}\n```',
        'Sure! Here you go: {"intents": []} Hope that helps.',
    ],
)
def test_extract_json_tolerates_wrapping(text):
    assert extract_json(text) == {"intents": []}


def test_extract_json_rejects_garbage():
    with pytest.raises(LLMError):
        extract_json("I think we should shuttle nitrogen.")


class FakeServer:
    """A tiny Ollama / OpenAI-compatible server that replies from a queue."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, **body})
                text = outer.replies.pop(0) if outer.replies else "{}"
                if self.path == "/api/chat":
                    out = {"message": {"content": text}, "prompt_eval_count": 10, "eval_count": 5}
                else:
                    out = {
                        "choices": [{"message": {"content": text}}],
                        "usage": {"prompt_tokens": 10},
                    }
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    servers = []

    def make(replies):
        s = FakeServer(replies)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def scripted(role, messages, context):
    return '{"intents": [], "source": "scripted"}'


@pytest.mark.parametrize("backend", ["ollama", "openai"])
def test_http_backends_parse_json(server, backend):
    srv = server(['```json\n{"intents": [{"kind": "set_trade_bias"}]}\n```'])
    route = ModelRoute(backend=backend, model="m", base_url=srv.url)
    router = Router({"mycelium": [route]})
    res = router.complete("mycelium", [{"role": "user", "content": "hi"}], want_json=True)
    assert res.parsed == {"intents": [{"kind": "set_trade_bias"}]}
    req = srv.requests[0]
    assert req["path"] == ("/api/chat" if backend == "ollama" else "/v1/chat/completions")
    assert (req.get("format") == "json") if backend == "ollama" else req["response_format"]


def test_retry_on_bad_output_then_succeed(server):
    srv = server(["not json at all", '{"ok": true}'])
    route = ModelRoute(backend="ollama", model="m", base_url=srv.url, retries=1)
    res = Router({"r": [route]}).complete("r", [], want_json=True)
    assert res.parsed == {"ok": True} and res.attempts == 2 and not res.fell_back


def test_falls_back_to_scripted_after_repeated_garbage(server):
    srv = server(["nope", "still nope"])
    route = ModelRoute(backend="ollama", model="m", base_url=srv.url, retries=1)
    router = Router(
        {"mycelium": [route, ModelRoute(backend="scripted")]}, scripted={"mycelium": scripted}
    )
    res = router.complete("mycelium", [], want_json=True)
    assert res.fell_back and res.parsed["source"] == "scripted"


def test_unreachable_server_is_marked_down_and_skipped():
    dead = ModelRoute(backend="ollama", model="m", base_url="http://127.0.0.1:9", timeout_s=2)
    router = Router(
        {"mycelium": [dead, ModelRoute(backend="scripted")]}, scripted={"mycelium": scripted}
    )
    for _ in range(2):
        assert router.complete("mycelium", [], want_json=True).fell_back
    assert dead.label in router._down


def test_all_routes_failing_raises():
    with pytest.raises(LLMError, match="all routes failed"):
        Router({"r": [ModelRoute(backend="scripted")]}).complete("r", [])


def test_routes_from_spec():
    r = routes_from_spec("ollama:qwen2.5:7b-instruct")
    assert (
        r["mycelium"][0].model == "qwen2.5:7b-instruct" and r["mycelium"][-1].backend == "scripted"
    )
    assert (
        routes_from_spec("openai:x", "http://gpu:8000")["narrator"][0].base_url == "http://gpu:8000"
    )
    with pytest.raises(ValueError):
        routes_from_spec("claude:big")
