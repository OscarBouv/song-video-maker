"""Analyze scenes using vision LLMs via OpenRouter."""
import base64
import json
from pathlib import Path

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_ANALYZER_MODEL,
    SCENE_ANALYSIS_BATCH,
)
from models import Scene
from pipeline import cache


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _system_prompt(film_name: str, n_frames: int, characters: list[str] | None = None) -> str:
    char_block = ""
    if characters:
        names = "\n".join(f"  - {c}" for c in characters)
        char_block = (
            f"\nKnown characters in '{film_name}' — identify them by name whenever they appear:\n"
            f"{names}\n"
            f"Always use these exact names in 'characters_present' and in the description.\n"
        )

    return (
        f"You are a cinematographer and film analyst examining scenes from '{film_name}'.\n"
        f"{char_block}\n"
        f"For each scene (labeled SCENE N) you receive {n_frames} sampled frames.\n\n"
        "Return a JSON array — one object per scene — with EXACTLY these keys:\n\n"

        '  "index": <scene index (integer)>,\n\n'

        '  "description": "<One vivid sentence describing WHAT IS HAPPENING: who is present, '
        "what they are doing, and the dramatic situation. Be specific and visual — "
        "prefer 'Alana stands frozen in the parking lot, watching Gary's truck disappear around the corner' "
        "over 'a woman stands outside'. Mention character names if visible.>,\n\n"

        '  "characters_present": [<list of character names visible in this scene — empty list [] if none>],\n\n'

        '  "emotion": "<The dominant emotional register of the scene in 2–5 evocative words. '
        "Think like a music-video director: what FEELING should a viewer walk away with? "
        "e.g. 'aching longing', 'quiet joy', 'electric tension', 'bittersweet nostalgia', 'defiant freedom'.>,\n\n"

        '  "shot_type": "<The dominant camera framing, choosing from: '
        "extreme close-up | close-up | medium close-up | medium | medium wide | wide | extreme wide | "
        "over-the-shoulder | two-shot | POV | insert/detail shot. "
        "If the scene cuts between framings, name the most emotionally powerful one.>,\n\n"

        '  "lighting": "<Describe the lighting quality and colour in 3–8 words. '
        "e.g. 'warm golden hour, soft shadows', 'harsh fluorescent, clinical', "
        "'blue-green neon, night', 'diffused overcast, flat', 'high contrast backlit silhouette'.>,\n\n"

        '  "visual_power": <integer 1–5 — how iconic and cinematically striking is this scene '
        "as a standalone music-video shot? Apply this rubric strictly:\n"
        "    5 = Showstopper: a frame you could print as a poster. Extraordinary light, composition,\n"
        "        or emotional charge. The hero shot of any music video.\n"
        "    4 = Strong: visually beautiful or emotionally arresting. Great light or framing.\n"
        "        A confident choice for a key moment in a fan edit.\n"
        "    3 = Solid: competent and clean, nothing wrong but nothing exceptional.\n"
        "    2 = Weak: flat light, cluttered background, or awkward composition.\n"
        "    1 = Poor: blurry, badly framed, heavily obstructed, or aesthetically unappealing.>,\n\n"

        f'  "is_film_related": <true/false — is this scene from \'{film_name}\'?>,\n\n'

        '  "is_aesthetic": <true/false — is this scene visually suitable for an Instagram Reel? '
        "Mark FALSE if any frame shows: overlaid text/subtitles/watermarks/channel logos, "
        "explicit or graphic content, heavy pixelation/corruption, black bars covering >30 % of the frame, "
        "or on-screen UI / interface elements>,\n\n"

        '  "confidence": <0.0–1.0 — your confidence that is_film_related is correct>\n\n'

        "Return ONLY the JSON array, no markdown, no commentary."
    )


def _build_content(scenes_batch: list[Scene], film_name: str, characters: list[str] | None = None) -> list[dict]:
    parts: list[dict] = [
        {"type": "text", "text": _system_prompt(film_name, len(scenes_batch[0].frames), characters)}
    ]
    for scene in scenes_batch:
        parts.append({"type": "text", "text": f"\n--- SCENE {scene.index} (duration: {scene.duration:.1f}s) ---"})
        for frame_path in scene.frames:
            if not Path(frame_path).exists():
                continue
            data = _encode_image(frame_path)
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{data}"},
            })
    return parts


def _call_openrouter(content: list[dict], model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content.strip()

def _apply_result(scene: Scene, r: dict) -> None:
    scene.description        = r.get("description", "")
    scene.characters_present = r.get("characters_present", [])
    scene.emotion            = r.get("emotion", "")
    scene.shot_type          = r.get("shot_type", "")
    scene.lighting           = r.get("lighting", "")
    scene.visual_power       = max(1, min(5, int(r.get("visual_power", 3))))  # clamp 1–5
    scene.is_film_related    = r.get("is_film_related", True)
    scene.is_aesthetic       = r.get("is_aesthetic", True)
    scene.confidence         = float(r.get("confidence", 1.0))


def _parse_json(raw: str, batch_num: int) -> list[dict] | None:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[scene_analyzer] Warning: failed to parse JSON for batch {batch_num}: {e}")
        print(f"Raw response: {raw[:500]}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_scenes(
    scenes: list[Scene],
    film_name: str,
    batch_size: int = SCENE_ANALYSIS_BATCH,
    model: str | None = None,
    characters: list[str] | None = None,
    max_batches: int | None = None,
) -> list[Scene]:
    """Send scenes to a vision LLM via OpenRouter in batches. Cache results by frame content hash.
    Re-running with the same video costs nothing — all scenes load from cache.

    Args:
        characters: Optional list of main character names (e.g. ["Alana Kane", "Gary Valentine"]).
                    When provided the vision prompt asks the model to name them in descriptions,
                    producing richer context for the matcher. Characters are included in the
                    cache key, so adding them busts any previous generic cache.
    """

    resolved_model = model or DEFAULT_ANALYZER_MODEL.value

    # Build a stable cache key that encodes character context so adding/removing
    # characters triggers a fresh analysis rather than returning stale generic descriptions.
    # The model is embedded in the cache *filename* (via cache.get_scene/set_scene),
    # so switching models always produces a cache miss — no need to put it in the key too.
    cache_context = film_name
    if characters:
        cache_context = f"{film_name}|chars:{','.join(sorted(characters))}"

    # ── Separate cached and uncached ─────────────────────────────────────────
    uncached: list[Scene] = []
    for scene in scenes:
        cached = cache.get_scene(scene.frames, cache_context, model=resolved_model)
        if cached:
            _apply_result(scene, cached)
        else:
            uncached.append(scene)

    n_cached = len(scenes) - len(uncached)
    if n_cached:
        print(f"[scene_analyzer] {n_cached}/{len(scenes)} scenes loaded from cache")
    if not uncached:
        usable = sum(1 for s in scenes if s.is_usable)
        print(f"[scene_analyzer] {usable}/{len(scenes)} usable scenes (all cached)")
        return scenes

    char_hint = f" | characters: {', '.join(characters)}" if characters else ""
    total_batches = (len(uncached) + batch_size - 1) // batch_size
    run_batches = min(total_batches, max_batches) if max_batches else total_batches
    limit_hint = f" (capped at {run_batches}/{total_batches} batches)" if max_batches and run_batches < total_batches else ""
    print(f"[scene_analyzer] Calling OpenRouter ({resolved_model}) for {len(uncached)} uncached scenes{char_hint}{limit_hint}...")

    # ── Process uncached scenes in batches ───────────────────────────────────
    for batch_start in range(0, len(uncached), batch_size):
        batch_num = batch_start // batch_size + 1
        if batch_num > run_batches:
            print(f"[scene_analyzer] Batch limit reached — stopping after {run_batches} batch(es)")
            break
        batch = uncached[batch_start : batch_start + batch_size]
        print(f"[scene_analyzer] Batch {batch_num}/{run_batches} ({len(batch)} scenes)...")

        content = _build_content(batch, film_name, characters)
        raw = _call_openrouter(content, resolved_model)

        results = _parse_json(raw, batch_num)
        if results is None:
            continue

        result_by_index = {r["index"]: r for r in results}
        for scene in batch:
            if scene.index in result_by_index:
                r = result_by_index[scene.index]
                _apply_result(scene, r)
                cache.set_scene(
                    scene.frames,
                    cache_context,
                    {
                        "description":        scene.description,
                        "characters_present": scene.characters_present,
                        "emotion":            scene.emotion,
                        "shot_type":          scene.shot_type,
                        "lighting":           scene.lighting,
                        "visual_power":       scene.visual_power,
                        "is_film_related":    scene.is_film_related,
                        "is_aesthetic":       scene.is_aesthetic,
                        "confidence":         scene.confidence,
                    },
                    model=resolved_model,
                )

    film_related = sum(1 for s in scenes if s.is_film_related)
    aesthetic    = sum(1 for s in scenes if s.is_aesthetic)
    usable       = sum(1 for s in scenes if s.is_usable)
    print(f"[scene_analyzer] {film_related}/{len(scenes)} film-related  |  {aesthetic}/{len(scenes)} aesthetic  |  {usable} usable")
    return scenes
