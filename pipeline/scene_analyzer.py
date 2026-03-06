"""Analyze scenes using Claude vision: describe content and filter by film."""
import base64
import json
from pathlib import Path

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, SCENE_ANALYSIS_BATCH
from models import Scene


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _build_batch_prompt(scenes_batch: list[Scene], film_name: str) -> list[dict]:
    """Build a Claude message for a batch of scenes."""
    content: list[dict] = []

    content.append({
        "type": "text",
        "text": (
            f"You are analyzing scenes from a video compilation to find clips from the film '{film_name}'.\n\n"
            f"For each scene below (labeled SCENE N), I will show you {len(scenes_batch[0].frames)} frames sampled from that scene.\n"
            "For each scene, respond with a JSON object containing:\n"
            '  "index": <scene index>,\n'
            '  "description": "<one sentence describing the scene visually>",\n'
            '  "is_film_related": <true/false — whether this scene appears to be from the film \'' + film_name + '\'>,\n'
            '  "confidence": <0.0-1.0 — your confidence in the is_film_related judgment>\n\n'
            "Return a JSON array of these objects, one per scene. Nothing else."
        ),
    })

    for scene in scenes_batch:
        content.append({
            "type": "text",
            "text": f"\n--- SCENE {scene.index} (duration: {scene.duration:.1f}s) ---",
        })
        for frame_path in scene.frames:
            if not Path(frame_path).exists():
                continue
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _encode_image(frame_path),
                },
            })

    return content


def analyze_scenes(
    scenes: list[Scene],
    film_name: str,
    batch_size: int = SCENE_ANALYSIS_BATCH,
) -> list[Scene]:
    """Send scenes to Claude vision in batches. Returns scenes with description,
    is_film_related, and confidence filled in."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    for batch_start in range(0, len(scenes), batch_size):
        batch = scenes[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(scenes) + batch_size - 1) // batch_size
        print(f"[scene_analyzer] Analyzing batch {batch_num}/{total_batches} ({len(batch)} scenes)...")

        content = _build_batch_prompt(batch, film_name)

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )

        raw = response.content[0].text.strip()

        # Parse JSON — Claude may wrap it in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            results = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[scene_analyzer] Warning: failed to parse JSON for batch {batch_num}: {e}")
            print(f"Raw response: {raw[:500]}")
            continue

        # Map results back to scenes by index
        result_by_index = {r["index"]: r for r in results}
        for scene in batch:
            if scene.index in result_by_index:
                r = result_by_index[scene.index]
                scene.description = r.get("description", "")
                scene.is_film_related = r.get("is_film_related", True)
                scene.confidence = float(r.get("confidence", 1.0))

    film_related = sum(1 for s in scenes if s.is_film_related)
    print(f"[scene_analyzer] {film_related}/{len(scenes)} scenes identified as from '{film_name}'")
    return scenes
