"""Disk-based cache for expensive API calls (scene analysis, LLM matching).

Cache files live in temp/cache/ and persist across runs.
Keys are content hashes so the same video/prompt always hits the same cache entry.

Each model gets its own cache file so switching models always triggers a fresh call:
  temp/cache/scene_analysis_<model_slug>.json
  temp/cache/llm_responses_<model_slug>.json

Each cache entry also stores a ``_meta`` block with the model name and creation
timestamp, making it easy to inspect the cache files and understand their provenance.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from config import TEMP_DIR

CACHE_DIR = TEMP_DIR / "cache"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _model_slug(model: str) -> str:
    """Convert a model ID like 'google/gemini-2.0-flash-001' to 'google_gemini_2_0_flash_001'."""
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def _cache_path(namespace: str, model: str = "") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if model:
        slug = _model_slug(model)
        return CACHE_DIR / f"{namespace}_{slug}.json"
    return CACHE_DIR / f"{namespace}.json"


def _load(namespace: str, model: str = "") -> dict:
    path = _cache_path(namespace, model)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save(namespace: str, data: dict, model: str = "") -> None:
    with open(_cache_path(namespace, model), "w") as f:
        json.dump(data, f, indent=2)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Hashing helpers ───────────────────────────────────────────────────────────

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
# File: temp/cache/scene_analysis_<model_slug>.json
# Entry structure:
#   "<hash>": {
#       "description": "...", "is_film_related": true, ...,   ← the actual data
#       "_meta": {"model": "...", "film": "...", "created_at": "..."}
#   }

def _scene_key(frame_paths: list[str], film_name: str) -> str:
    return _hash_files(frame_paths) + ":" + _hash_str(film_name)


def get_scene(frame_paths: list[str], film_name: str, model: str = "") -> dict | None:
    """Return cached scene analysis result dict, or None if not cached."""
    entry = _load("scene_analysis", model).get(_scene_key(frame_paths, film_name))
    if entry is None:
        return None
    # Strip internal _meta before returning so callers see only the data fields
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def set_scene(frame_paths: list[str], film_name: str, result: dict, model: str = "") -> None:
    """Persist a scene analysis result, attaching model + timestamp metadata."""
    cache = _load("scene_analysis", model)
    cache[_scene_key(frame_paths, film_name)] = {
        **result,
        "_meta": {
            "model": model,
            "film": film_name,
            "created_at": _now_iso(),
        },
    }
    _save("scene_analysis", cache, model)


# ── LLM response cache ────────────────────────────────────────────────────────
# File: temp/cache/llm_responses_<model_slug>.json
# Entry structure:
#   "<hash>": {
#       "response": "...",   ← the raw LLM text
#       "_meta": {"model": "...", "provider": "...", "created_at": "..."}
#   }

def _llm_key(prompt: str, model: str, provider: str) -> str:
    return _hash_str(f"{provider}:{model}:{prompt}")


def get_llm(prompt: str, model: str, provider: str) -> str | None:
    """Return a cached raw LLM response string, or None if not cached."""
    entry = _load("llm_responses", model).get(_llm_key(prompt, model, provider))
    if entry is None:
        return None
    # Support both new wrapped format {"response": "..."} and legacy plain string
    if isinstance(entry, dict):
        return entry.get("response")
    return entry  # legacy plain string


def set_llm(prompt: str, model: str, provider: str, response: str) -> None:
    """Persist a raw LLM response string with model/provider metadata."""
    cache = _load("llm_responses", model)
    cache[_llm_key(prompt, model, provider)] = {
        "response": response,
        "_meta": {
            "model": model,
            "provider": provider,
            "created_at": _now_iso(),
        },
    }
    _save("llm_responses", cache, model)
