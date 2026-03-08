"""Match scenes to lyric lines using an LLM via OpenRouter. Produces a reviewable plan file."""
import json
from pathlib import Path

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_MATCHER_MODEL,
    MAX_REEL_DURATION,
)
from models import Scene, LyricLine, MatchedSegment
from pipeline import cache


def _call_openrouter(prompt: str, model: str) -> str:
    cached = cache.get_llm(prompt, model, "openrouter")
    if cached:
        print("[matcher] Using cached OpenRouter response")
        return cached
    from openai import OpenAI
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    result = response.choices[0].message.content.strip()
    cache.set_llm(prompt, model, "openrouter", result)
    return result


def generate_plan(
    scenes: list[Scene],
    lyrics: list[LyricLine],
    film_name: str,
    song_name: str,
    output_dir: Path,
    slug: str = "output",
    model: str | None = None,
) -> list[MatchedSegment]:
    """Ask an LLM via OpenRouter to propose a scene-to-lyric mapping.

    Args:
        model: Model ID override. Defaults to DEFAULT_MATCHER_MODEL.

    Saves two files:
      - {output_dir}/{slug}_plan.json   (machine-readable, user can edit)
      - {output_dir}/{slug}_plan_readable.txt  (human-friendly preview)

    Returns the MatchedSegment list so the caller can proceed directly if desired.
    """
    film_scenes = [s for s in scenes if s.is_usable]
    if not film_scenes:
        print("[matcher] Warning: no usable scenes found — using all scenes")
        film_scenes = scenes

    prompt = _build_prompt(film_scenes, lyrics, film_name, song_name)

    resolved_model = model or DEFAULT_MATCHER_MODEL.value
    print(f"[matcher] Calling OpenRouter ({resolved_model}) for scene-to-lyric plan...")
    raw = _call_openrouter(prompt, resolved_model)

    # Strip markdown code fences if present
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        plan_data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude returned invalid JSON for plan: {e}\nRaw:\n{raw[:1000]}")

    # Build MatchedSegment list
    scene_by_index = {s.index: s for s in scenes}
    segments: list[MatchedSegment] = []

    for entry in plan_data:
        scene_idx = entry["scene_index"]
        if scene_idx not in scene_by_index:
            print(f"[matcher] Warning: plan references unknown scene index {scene_idx}")
            continue

        lyric_indices = entry.get("lyric_indices", [])
        matched_lyrics = [lyrics[i] for i in lyric_indices if 0 <= i < len(lyrics)]

        segments.append(
            MatchedSegment(
                scene_index=scene_idx,
                lyric_lines=matched_lyrics,
                scene_trim_start=entry.get("scene_trim_start", 0.0),
                scene_trim_end=entry.get("scene_trim_end", -1.0),
            )
        )

    # Save JSON plan
    plan_path = output_dir / f"{slug}_plan.json"
    with open(plan_path, "w") as f:
        json.dump([s.to_dict() for s in segments], f, indent=2)

    # Save human-readable plan
    readable_path = output_dir / f"{slug}_plan_readable.txt"
    _write_readable_plan(segments, scene_by_index, film_name, song_name, readable_path)

    print(f"\n[matcher] Plan saved:")
    print(f"  Machine-readable : {plan_path}")
    print(f"  Human-readable   : {readable_path}")
    print(
        "\n→ Review the readable plan above. Edit the JSON plan if needed,\n"
        "  then re-run with --render-only to render the final video.\n"
    )

    return segments


def load_plan(plan_path: Path, lyrics: list[LyricLine]) -> list[MatchedSegment]:
    """Load a previously saved plan.json back into MatchedSegment objects."""
    with open(plan_path) as f:
        data = json.load(f)
    return [MatchedSegment.from_dict(entry) for entry in data]


def _build_prompt(
    scenes: list[Scene], lyrics: list[LyricLine], film_name: str, song_name: str
) -> str:
    scene_lines = "\n".join(
        f"  Scene {s.index} [{s.start_time:.1f}s-{s.end_time:.1f}s, {s.duration:.1f}s]: {s.description}"
        for s in scenes
    )
    lyric_lines = "\n".join(
        f"  [{i}] [{l.start_time:.2f}s-{l.end_time:.2f}s] {l.text}"
        for i, l in enumerate(lyrics)
    )
    total_lyric_duration = lyrics[-1].end_time if lyrics else 0
    max_dur = min(MAX_REEL_DURATION, total_lyric_duration)

    return f"""You are a video editor creating an aesthetic Instagram Reel from the film '{film_name}' set to the song '{song_name}'.

AVAILABLE SCENES from the film:
{scene_lines}

SONG LYRICS with timestamps (0 = song start):
{lyric_lines}

TASK:
Create a scene-to-lyric plan for a ~{max_dur:.0f}s video. Assign each lyric line to the most emotionally/visually fitting scene. You may reuse scenes or skip scenes, but cover all lyric lines. Each scene segment should be 1-5 seconds.

Return a JSON array where each element has:
  "scene_index": <integer — must be one of the available scene indices above>,
  "lyric_indices": [<list of lyric line indices [0..{len(lyrics)-1}] that play during this scene>],
  "scene_trim_start": <float — seconds into the scene to start from (default 0.0)>,
  "scene_trim_end": <float — seconds into the scene to end at (-1 means end of scene)>,
  "reasoning": "<one sentence explaining why this scene fits these lyrics>"

Important:
- Cover all {len(lyrics)} lyric lines in order
- Keep total video duration under {max_dur:.0f} seconds
- Match scene mood/energy to lyric mood
- Return only valid JSON, no extra text"""


def _write_readable_plan(
    segments: list[MatchedSegment],
    scene_by_index: dict[int, Scene],
    film_name: str,
    song_name: str,
    out_path: Path,
) -> None:
    lines = [
        f"PLAN: '{film_name}' × '{song_name}'",
        "=" * 60,
        "",
    ]
    total = 0.0
    for i, seg in enumerate(segments, 1):
        scene = scene_by_index.get(seg.scene_index)
        scene_desc = scene.description if scene else "unknown"
        scene_start = scene.start_time if scene else 0
        scene_end = scene.end_time if scene else 0

        trim_start = seg.scene_trim_start
        trim_end = seg.scene_trim_end if seg.scene_trim_end >= 0 else (scene_end - scene_start)
        clip_dur = trim_end - trim_start
        total += clip_dur

        lines.append(f"Segment {i}  (clip duration: {clip_dur:.1f}s | running total: {total:.1f}s)")
        lines.append(f"  Scene {seg.scene_index} [{scene_start:.1f}s-{scene_end:.1f}s in source]")
        lines.append(f"  Trim: {trim_start:.1f}s → {trim_end:.1f}s within scene")
        lines.append(f"  Visual: {scene_desc}")
        lines.append(f"  Lyrics:")
        for lyric in seg.lyric_lines:
            lines.append(f"    [{lyric.start_time:.2f}s] {lyric.text}")
        lines.append("")

    lines.append(f"Total estimated duration: {total:.1f}s")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
