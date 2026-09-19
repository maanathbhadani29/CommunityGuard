"""Ollama HTTP interface with complete request/response logging.

Uses the documented /api/chat and /api/tags endpoints. It does not execute any
LLM-generated code. Network failure aborts the run; it is never fabricated as an
LLM response. Malformed model text is logged and receives the notebook fallback.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)

def parse_output(raw: str, stage: str, profile="current") -> tuple[dict, str | None]:
    if profile not in ("current", "archived"):
        raise ValueError("Unknown prompt profile.")
    current_defense = stage != "forensics" and profile == "current"
    fallback = ({"diagnosis": "Unknown", "reasoning": "Error parsing."}
                if stage == "forensics" else
                ({"protocol": "DIGITAL_TWIN_OVERRIDE", "strategy": "Isolate compromised systems.",
                  "justification": "Fallback protocol."} if current_defense else
                 {"command": "DIGITAL_TWIN_OVERRIDE", "justification": "Fallback protocol."}))
    fields = tuple(fallback)

    def canonical(obj):
        # The verifier consumes only this protocol alias; strategy text is never executed.
        if current_defense:
            obj["command"] = obj["protocol"]
        return obj

    try:
        content = raw.strip()
        if profile == "current":
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
        obj = json.loads(content.strip())
        if not isinstance(obj, dict):
            raise ValueError("Expected a JSON object.")
        if current_defense and "justification" not in obj:
            for alias in ("reasoning", "rationale"):
                if alias in obj:
                    obj["justification"] = obj[alias]
                    break
        for f in fields:
            if f in obj and not isinstance(obj[f], str):
                raise ValueError(f"Field {f} must be a string.")
        missing = [f for f in fields if f not in obj]
        error = "Missing fields received defaults: " + ", ".join(missing) if missing else None
        return canonical({f: obj.get(f, fallback[f]) for f in fields}), error
    except (ValueError, TypeError) as exc:
        return canonical(fallback.copy()), str(exc)

class OllamaClient:
    def __init__(self, url="http://127.0.0.1:11434", model="llama3", seed=42,
                 temperature=0.1, timeout=300.0, json_mode=False, prompt_profile="current"):
        self.url, self.model = url.rstrip("/"), model
        self.seed, self.temperature, self.timeout = seed, temperature, timeout
        self.json_mode = json_mode
        self.prompt_profile = prompt_profile

    def _request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(self.url + path, data=body,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Ollama request failed at {self.url}{path}: {exc}. "
                               "Start Ollama and run: ollama pull llama3") from exc

    def preflight(self):
        tags = self._request("/api/tags")
        allowed = {self.model, self.model + ":latest"}
        models = [m for m in tags.get("models", []) if m.get("name") in allowed or m.get("model") in allowed]
        if not models:
            raise RuntimeError(f"Ollama model {self.model!r} is not installed. Run: ollama pull {self.model}")
        return {"url": self.url, "selected_model": models[0], "temperature": self.temperature,
                "seed": self.seed, "json_mode": self.json_mode, "prompt_profile": self.prompt_profile}

    def invoke(self, prompt, stage, log_path):
        request = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "options": {"temperature": self.temperature, "seed": self.seed}}
        if self.json_mode:
            request["format"] = "json"
        started = time.perf_counter()
        try:
            response = self._request("/api/chat", request)
        except Exception as exc:
            write_json(log_path, {"source": "ollama_http_error", "stage": stage,
                                 "request": request, "error": str(exc)})
            raise
        elapsed = time.perf_counter() - started
        raw = response.get("message", {}).get("content")
        if not isinstance(raw, str):
            write_json(log_path, {"source": "ollama_invalid_response", "stage": stage,
                                 "request": request, "response": response})
            raise RuntimeError("Ollama response has no text content; raw response was logged.")
        parsed, error = parse_output(raw, stage, self.prompt_profile)
        token_fields = (response.get("prompt_eval_count"), response.get("eval_count"))
        tokens = sum(token_fields) if all(isinstance(v, int) for v in token_fields) else None
        record = {"source": "ollama_live", "stage": stage, "prompt_profile": self.prompt_profile,
                  "request": request,
                  "response": response, "parsed": parsed, "parse_error": error,
                  "fallback_used": error is not None, "latency_seconds": elapsed,
                  "prompt_tokens": token_fields[0], "output_tokens": token_fields[1],
                  "total_tokens": tokens}
        write_json(log_path, record)
        return record
