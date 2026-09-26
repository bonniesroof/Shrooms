"""Model routing: local LLMs (Ollama, vLLM) with fallback chains, or scripted stand-ins.

Each agent role maps to an ordered list of routes. The router tries them in
order: a route that errors or times out after its retries falls through to the
next, and every attempt is logged. The `scripted` backend is a deterministic,
rule-based policy with the same interface, used for tests, CI, and as the
fallback of last resort, so a dead model server never stalls a run.

    ollama   native /api/chat with format=json when JSON is required
    openai   any OpenAI-compatible /v1/chat/completions (vLLM, llama.cpp, LM Studio)
    scripted a Python policy function (no network)
"""

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("shrooms.llm")

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": ...}
ScriptedPolicy = Callable[[str, list[Message], dict], str]


class LLMError(RuntimeError):
    pass


class ModelRoute(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Literal["ollama", "openai", "scripted"]
    model: str = "scripted"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    timeout_s: float = Field(120.0, gt=0.0)
    retries: int = Field(1, ge=0)  # extra attempts on bad output before falling through
    max_tokens: int = Field(800, gt=0)

    @property
    def label(self) -> str:
        return f"{self.backend}:{self.model}"


# Defaults sized for one ~7B model on a 10 GB GPU: both roles share it so Ollama
# never swaps models, and the scripted policy catches any failure.
LOCAL_7B = ModelRoute(backend="ollama", model="qwen2.5:7b-instruct")
SCRIPTED = ModelRoute(backend="scripted")
DEFAULT_ROUTES = {"mycelium": [LOCAL_7B, SCRIPTED], "narrator": [LOCAL_7B, SCRIPTED]}


@dataclass
class LLMResult:
    text: str
    route: ModelRoute
    latency_s: float
    attempts: int
    fell_back: bool
    parsed: dict | list | None = None
    usage: dict = field(default_factory=dict)


def extract_json(text: str) -> dict | list:
    """Parse JSON from model output, tolerating code fences and chatter around it."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"no JSON object found in model output: {text[:200]!r}")


class Router:
    def __init__(
        self,
        routes: dict[str, list[ModelRoute]] | None = None,
        scripted: dict[str, ScriptedPolicy] | None = None,
        client: httpx.Client | None = None,
    ):
        self.routes = routes or DEFAULT_ROUTES
        self.scripted = scripted or {}
        self.client = client or httpx.Client()
        self.calls: list[dict] = []  # per-call telemetry, for the run summary
        self._down: set[str] = set()  # routes that failed to connect; skipped thereafter

    def complete(
        self,
        role: str,
        messages: list[Message],
        *,
        context: dict | None = None,
        want_json: bool = False,
    ) -> LLMResult:
        chain = self.routes.get(role)
        if not chain:
            raise LLMError(f"no routes configured for role {role!r}")
        errors = []
        for position, route in enumerate(chain):
            if route.label in self._down:
                log.debug("%s: skipping %s (marked down earlier)", role, route.label)
                continue
            for attempt in range(1, route.retries + 2):
                start = time.perf_counter()
                try:
                    text, usage = self._call(route, role, messages, context or {}, want_json)
                    parsed = extract_json(text) if want_json else None
                except httpx.ConnectError as err:
                    self._down.add(route.label)
                    errors.append(f"{route.label}: cannot connect ({err})")
                    log.warning(
                        "%s: %s unreachable at %s; falling back", role, route.label, route.base_url
                    )
                    break
                except (httpx.HTTPError, LLMError, KeyError, ValueError) as err:
                    errors.append(f"{route.label} attempt {attempt}: {err}")
                    log.warning("%s: %s attempt %d failed: %s", role, route.label, attempt, err)
                    continue
                latency = time.perf_counter() - start
                result = LLMResult(text, route, latency, attempt, position > 0, parsed, usage)
                self.calls.append(
                    {
                        "role": role,
                        "route": route.label,
                        "latency_s": latency,
                        "attempts": attempt,
                        "fell_back": position > 0,
                        **usage,
                    }
                )
                log.debug(
                    "%s: %s answered in %.2fs (attempt %d)%s",
                    role,
                    route.label,
                    latency,
                    attempt,
                    " [FALLBACK]" if position else "",
                )
                log.debug("%s: raw response: %s", role, json.dumps(text))
                return result
        raise LLMError(f"all routes failed for {role}: " + " | ".join(errors))

    def _call(self, route, role, messages, context, want_json) -> tuple[str, dict]:
        if route.backend == "scripted":
            policy = self.scripted.get(role)
            if policy is None:
                raise LLMError(f"no scripted policy registered for role {role!r}")
            return policy(role, messages, context), {}
        log.debug(
            "%s: -> %s (%d messages, %d chars)",
            role,
            route.label,
            len(messages),
            sum(len(m["content"]) for m in messages),
        )
        if route.backend == "ollama":
            body = {
                "model": route.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": route.temperature, "num_predict": route.max_tokens},
            }
            if want_json:
                body["format"] = "json"
            r = self.client.post(f"{route.base_url}/api/chat", json=body, timeout=route.timeout_s)
            r.raise_for_status()
            data = r.json()
            usage = {
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
            }
            return data["message"]["content"], usage
        body = {
            "model": route.model,
            "messages": messages,
            "temperature": route.temperature,
            "max_tokens": route.max_tokens,
        }
        if want_json:
            body["response_format"] = {"type": "json_object"}
        r = self.client.post(
            f"{route.base_url}/v1/chat/completions", json=body, timeout=route.timeout_s
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"], data.get("usage", {})


def routes_from_spec(spec: str, base_url: str | None = None) -> dict[str, list[ModelRoute]]:
    """Build routes from a CLI spec like 'ollama:qwen2.5:7b-instruct' or 'scripted'.

    Every chain ends with the scripted fallback.
    """
    if spec == "scripted":
        return {"mycelium": [SCRIPTED], "narrator": [SCRIPTED]}
    backend, _, model = spec.partition(":")
    if backend not in ("ollama", "openai") or not model:
        raise ValueError(f"bad model spec {spec!r}; use scripted, ollama:<model> or openai:<model>")
    default_url = "http://localhost:11434" if backend == "ollama" else "http://localhost:8000"
    route = ModelRoute(backend=backend, model=model, base_url=base_url or default_url)
    return {"mycelium": [route, SCRIPTED], "narrator": [route, SCRIPTED]}
