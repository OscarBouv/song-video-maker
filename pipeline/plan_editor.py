"""LLM-based interactive plan editor for targeted clip-level adjustments.

Applies natural-language instructions to an existing plan, producing a modified
plan ready for re-render.  Uses a cheap, fast model (Gemini Flash by default)
since the task is targeted JSON editing, not creative generation.

What CAN be changed per clip:
  - scene_index       — swap to a different scene
  - scene_trim_start  — where in the scene to start (framing)
  - scene_trim_end is recomputed automatically to preserve clip duration

What CANNOT be changed:
  - song_start / song_end  — audio slot positions are fixed by lyric timestamps
  - Adding or removing clips — slot structure is preserved

Lyric subtitle text corrections go directly to lyrics.json (passed back to caller).
"""
import json
from pathlib import Path

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OpenRouterModel,
)
from models import Scene, LyricLine, MatchedSegment

# Cheap + reliable JSON editing — fractions of a cent per call
DEFAULT_EDIT_MODEL = OpenRouterModel.gemini_flash


# ── Public API ────────────────────────────────────────────────────────────────

def edit_plan(
    segments: list[MatchedSegment],
    scenes: list[Scene],
    lyrics: list[LyricLine],
    instruction: str,
    film_name: str,
    song_name: str,
    model: str | None = None,
    ollama_url: str | None = None,
) -> tuple[list[MatchedSegment], list[LyricLine]]:
    """Apply a natural-language editing instruction to the plan.

    Args:
        ollama_url: If set, use local Ollama at this base URL instead of OpenRouter.
                    The model name is taken from the ``model`` arg (e.g. 'qwen2.5:14b').

    Returns:
        (modified_segments, modified_lyrics) — caller is responsible for saving.
    """
    resolved_model = model or DEFAULT_EDIT_MODEL.value
    prompt = _build_edit_prompt(segments, scenes, lyrics, instruction, film_name, song_name)

    provider = f"Ollama @ {ollama_url}" if ollama_url else f"OpenRouter"
    print(f"[plan_editor] Sending instruction to {provider} ({resolved_model})...")

    raw = _call_llm(prompt, resolved_model, ollama_url)

    # Strip markdown fences if present
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        diff = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"[plan_editor] LLM returned invalid JSON: {e}\nRaw response:\n{raw[:600]}"
        )

    scene_by_index = {s.index: s for s in scenes}
    new_segments = _apply_clip_changes(
        segments, diff.get("clip_changes", []), scene_by_index
    )
    new_lyrics = _apply_lyric_changes(lyrics, diff.get("lyric_changes", []))
    return new_segments, new_lyrics


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_edit_prompt(
    segments: list[MatchedSegment],
    scenes: list[Scene],
    lyrics: list[LyricLine],
    instruction: str,
    film_name: str,
    song_name: str,
) -> str:
    _STAR  = "★"
    _EMPTY = "☆"

    # ── Scene catalogue ───────────────────────────────────────────────────────
    scene_lines = []
    for s in sorted(scenes, key=lambda x: x.index):
        if not (s.is_film_related or s.is_aesthetic):
            continue  # skip unusable scenes to keep prompt compact
        vp    = max(1, min(5, s.visual_power))
        stars = _STAR * vp + _EMPTY * (5 - vp)
        chars = ", ".join(s.characters_present) if s.characters_present else "—"
        scene_lines.append(
            f"  Sc {s.index:>3}  [{s.start_time:.1f}s–{s.end_time:.1f}s, {s.duration:.1f}s]"
            f"  {stars}  {s.emotion or '—'}  |  chars: {chars}"
            f"\n          {s.description or '(no description)'}"
        )
    scenes_block = "\n".join(scene_lines)

    # ── Current plan ──────────────────────────────────────────────────────────
    plan_lines = []
    scene_by_index = {s.index: s for s in scenes}
    for i, seg in enumerate(segments, 1):
        scene = scene_by_index.get(seg.scene_index)
        # Clip duration = audio slot length (song_start/song_end define the slot)
        if seg.song_start >= 0:
            clip_dur = seg.song_end - seg.song_start
        else:
            clip_dur = max(0.0, seg.scene_trim_end - seg.scene_trim_start)
        vp    = scene.visual_power if scene else 3
        stars = _STAR * vp + _EMPTY * (5 - vp)
        chars = ", ".join(scene.characters_present) if scene and scene.characters_present else "—"
        desc  = (scene.description or "unknown")[:90]
        lyric_texts = [ll.text for ll in seg.lyric_lines if ll.end_time > ll.start_time]
        lyric_str   = " | ".join(f'"{t}"' for t in lyric_texts) if lyric_texts else "(silence/gap)"
        plan_lines.append(
            f"  Clip {i:>2}  [{seg.song_start:.2f}s–{seg.song_end:.2f}s, {clip_dur:.2f}s]"
            f"  Scene {seg.scene_index:>3} {stars}"
            f"  trim: {seg.scene_trim_start:.2f}–{seg.scene_trim_end:.2f}"
            f"\n          Chars: {chars}  |  Emotion: {scene.emotion if scene else '—'}"
            f"\n          Visual: {desc}"
            f"\n          Lyric: {lyric_str}"
        )
    plan_block = "\n".join(plan_lines)

    # ── Lyrics index (for text corrections) ──────────────────────────────────
    lyric_lines = []
    for i, ll in enumerate(lyrics):
        if ll.end_time > ll.start_time:
            lyric_lines.append(f"  [{i}]  {ll.start_time:.2f}s–{ll.end_time:.2f}s  \"{ll.text}\"")
    lyrics_block = "\n".join(lyric_lines)

    n_usable = sum(1 for s in scenes if s.is_film_related or s.is_aesthetic)

    return f"""You are a film editor making targeted adjustments to a music video plan.
Film: '{film_name}'  |  Song: '{song_name}'

{'═' * 70}
AVAILABLE SCENES  ({n_usable} usable scenes shown — others filtered)
{'═' * 70}
{scenes_block}

{'═' * 70}
CURRENT PLAN  ({len(segments)} clips)
{'═' * 70}
{plan_block}

{'═' * 70}
LYRIC LINES  (for subtitle text corrections only)
{'═' * 70}
{lyrics_block}

{'═' * 70}
EDITING INSTRUCTION
{'═' * 70}
{instruction}

{'═' * 70}
RULES
{'═' * 70}
• Only include clips that need changing — omit unchanged clips.
• When swapping a scene: pick one whose duration ≥ the clip's duration (shown in brackets).
• scene_trim_start must be ≥ 0 and ≤ (scene_duration − clip_duration).
• song_start/song_end are FIXED — the audio slot positions cannot change.
• lyric_changes corrects subtitle display text only — does not shift timing.

Return ONLY valid JSON, no markdown, no extra text:
{{
  "clip_changes": [
    {{
      "clip_number": <1-indexed integer matching plan above>,
      "scene_index": <integer — must exist in Available Scenes>,
      "scene_trim_start": <float — seconds into the scene where clip starts>,
      "reasoning": "<one sentence: why this scene fits better>"
    }}
  ],
  "lyric_changes": [
    {{
      "lyric_index": <integer — from Lyric Lines index above>,
      "text": "<corrected subtitle text>"
    }}
  ]
}}

Both arrays may be empty [] if that type of change is not needed.
If you only need clip changes, return "lyric_changes": [].
If you only need lyric corrections, return "clip_changes": []."""


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, model: str, ollama_url: str | None = None) -> str:
    """Call via OpenAI-compatible API — either OpenRouter or local Ollama."""
    from openai import OpenAI

    if ollama_url:
        base = ollama_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        client = OpenAI(api_key="ollama", base_url=base)
    else:
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ── Change application ────────────────────────────────────────────────────────

def _apply_clip_changes(
    segments: list[MatchedSegment],
    changes: list[dict],
    scene_by_index: dict[int, Scene],
) -> list[MatchedSegment]:
    if not changes:
        return segments

    new_segments = list(segments)  # shallow copy; we replace entries in-place
    applied = 0

    for change in changes:
        clip_num = change.get("clip_number")
        if not isinstance(clip_num, int) or not (1 <= clip_num <= len(new_segments)):
            print(f"[plan_editor]   ⚠  invalid clip_number {clip_num!r} — skipping")
            continue

        seg = new_segments[clip_num - 1]

        # Preserve audio slot duration as clip length
        if seg.song_start >= 0:
            clip_duration = seg.song_end - seg.song_start
        else:
            clip_duration = max(0.01, seg.scene_trim_end - seg.scene_trim_start)

        new_scene_idx = int(change.get("scene_index", seg.scene_index))
        # If no explicit trim_start given and we're swapping scenes, default to 0
        if "scene_trim_start" in change:
            new_trim_start = float(change["scene_trim_start"])
        elif new_scene_idx != seg.scene_index:
            new_trim_start = 0.0
        else:
            new_trim_start = seg.scene_trim_start

        if new_scene_idx not in scene_by_index:
            print(f"[plan_editor]   ⚠  scene_index {new_scene_idx} not found — skipping clip {clip_num}")
            continue

        scene = scene_by_index[new_scene_idx]
        scene_dur = scene.end_time - scene.start_time

        # Clamp trim_start so the clip fits within the scene
        max_trim_start = max(0.0, scene_dur - clip_duration)
        new_trim_start = max(0.0, min(new_trim_start, max_trim_start))
        new_trim_end   = new_trim_start + clip_duration

        reasoning = change.get("reasoning", "")
        arrow = f"→ Sc {new_scene_idx}" if new_scene_idx != seg.scene_index else f"  Sc {seg.scene_index}"
        print(
            f"[plan_editor]   Clip {clip_num:>2}: {arrow}"
            f"  trim {seg.scene_trim_start:.2f}→{new_trim_start:.2f}"
            + (f"  | {reasoning}" if reasoning else "")
        )

        new_segments[clip_num - 1] = MatchedSegment(
            scene_index=new_scene_idx,
            lyric_lines=seg.lyric_lines,
            scene_trim_start=new_trim_start,
            scene_trim_end=new_trim_end,
            song_start=seg.song_start,
            song_end=seg.song_end,
        )
        applied += 1

    print(f"[plan_editor] {applied}/{len(changes)} clip change(s) applied")
    return new_segments


def _apply_lyric_changes(
    lyrics: list[LyricLine],
    changes: list[dict],
) -> list[LyricLine]:
    if not changes:
        return lyrics

    new_lyrics = list(lyrics)
    applied = 0

    for change in changes:
        idx  = change.get("lyric_index")
        text = change.get("text", "").strip()
        if not isinstance(idx, int) or not (0 <= idx < len(new_lyrics)) or not text:
            print(f"[plan_editor]   ⚠  invalid lyric_change {change!r} — skipping")
            continue
        old = new_lyrics[idx]
        new_lyrics[idx] = LyricLine(text=text, start_time=old.start_time, end_time=old.end_time)
        print(f"[plan_editor]   Lyric [{idx}]: \"{old.text}\" → \"{text}\"")
        applied += 1

    print(f"[plan_editor] {applied}/{len(changes)} lyric change(s) applied")
    return new_lyrics
