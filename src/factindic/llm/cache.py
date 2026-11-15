"""Persistent SHA-256 hashed response cache for LLM calls."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class ResponseCache:
    """Caches LLM inputs and responses to ensure determinism and eliminate API costs on re-runs."""

    def __init__(self, cache_file: str = "data/cache/llm_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _hash_key(self, prompt: str, text: str, model: str) -> str:
        payload = f"{model}:::{prompt}:::{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def save(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def get(self, prompt: str, text: str, model: str) -> Optional[str]:
        key = self._hash_key(prompt, text, model)
        if key in self.cache:
            return self.cache[key].get("response")
        return None

    def put(self, prompt: str, text: str, model: str, response: str, latency_ms: int = 0) -> None:
        key = self._hash_key(prompt, text, model)
        self.cache[key] = {
            "response": response,
            "latency_ms": latency_ms,
            "prompt_hash": key[:12],
        }
        self.save()
