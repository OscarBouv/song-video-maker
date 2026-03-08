"""Disk-based cache for expensive API calls (scene analysis, LLM matching).

Cache files live in temp/cache/ and persist across runs.
Keys are content hashes so the same video/prompt always hits the same cache entry.
"""
import hashlib
import json
from pathlib import Path

from config import TEMP_DIR

CACHE_DIR = TEMP_DIR / "cache"


def _cache_path(namespace: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{namespace}.json"


def _load(namespace: str) -> dict:
    path = _cache_path(namespace)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save(namespace: str, data: dict) -> None:
    with open(_cache_path(namespace), "w") as f:
        json.dump(data, f, indent=2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_files(paths: list[str]) -> str:
    """MD5 of concatenated file contents. Deterministic across runs."""
    h = hashlib.md5()
    for p in sorted(paths):
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except FileNotFoundError:
            pass
    return h.hexdigest()


def _hash_str(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


# ── Scene analysis cache ──────────────────────────────────────────────────────

def _scene_key(frame_paths: list[str], film_name: str) -> str:
    return _hash_files(frame_paths) + ":" + _hash_str(film_name)


def get_scene(frame_paths: list[str], film_name: str) -> dict | None:
    """Return cached scene analysis result, or None if not cached."""
    return _load("scene_analysis").get(_scene_key(frame_paths, film_name))


def set_scene(frame_paths: list[str], film_name: str, result: dict) -> None:
    """Persist a scene analysis result."""
    cache = _load("scene_analysis")
    cache[_scene_key(frame_paths, film_name)] = result
    _save("scene_analysis", cache)


# ── LLM response cache ────────────────────────────────────────────────────────

def _llm_key(prompt: str, model: str, provider: str) -> str:
    return _hash_str(f"{provider}:{model}:{prompt}")


def get_llm(prompt: str, model: str, provider: str) -> str | None:
    """Return a cached raw LLM response string, or None if not cached."""
    return _load("llm_responses").get(_llm_key(prompt, model, provider))


def set_llm(prompt: str, model: str, provider: str, response: str) -> None:
    """Persist a raw LLM response string."""
    cache = _load("llm_responses")
    cache[_llm_key(prompt, model, provider)] = response
    _save("llm_responses", cache)
