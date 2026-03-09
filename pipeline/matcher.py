"""Match scenes to lyric lines using an LLM via OpenRouter. Produces a reviewable plan file.

The plan is built around the FIXED lyric timeline — every slot's duration is immutable.
The LLM's only job is to select which scene (and where within it) to show in each slot.
"""
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_MATCHER_MODEL,
    MAX_REEL_DURATION,
    FFPROBE_BIN,
)
from models import Scene, LyricLine, MatchedSegment
from pipeline import cache


# Silence gaps shorter than this (in seconds) are merged into adjacent lyric slots
# rather than becoming their own visual slot.
_MIN_GAP_FOR_SLOT = 0.3

# Lyric slots longer than this get flagged as multi-clip candidates in the prompt
_MULTICLIP_THRESHOLD = 5.0


# ── ffprobe helper ────────────────────────────────────────────────────────────

def _probe_audio_duration(audio_path: Path) -> float:
    """Return the duration of an audio file in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                FFPROBE_BIN, "-v", "quiet",
                "-print_format", "json",
                "-show_entries", "format=duration",
                str(audio_path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception as e:
        print(f"[matcher] Warning: could not probe audio duration ({e})")
        return 0.0


# ── OpenRouter call ───────────────────────────────────────────────────────────

def _call_openrouter(prompt: str, model: str) -> str:
    cached = cache.get_llm(prompt, model, "openrouter")
    if cached:
        print("[matcher] Using cached OpenRouter response")
        return cached
    from openai import OpenAI
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    response = client.chat.completions.create(
        model=model,
        max_tokens=8192,   # plan JSON can be large: ~250 tok/entry × 30+ entries = 7500+ tok
        messages=[{"role": "user", "content": prompt}],
    )
    choice        = response.choices[0]
    finish_reason = choice.finish_reason or "unknown"
    result        = choice.message.content.strip()

    if finish_reason == "length":
        # Response was cut off — the cached truncated JSON would silently produce a
        # short plan on every future run.  Warn loudly; do NOT cache the truncated reply.
        print(
            f"[matcher] ⚠️  LLM response hit the token limit (finish_reason='length'). "
            f"The plan will be INCOMPLETE. Try a smaller song window or a faster model."
        )
        return result   # return but skip caching so a retry gets a fresh call

    print(f"[matcher] LLM response complete (finish_reason='{finish_reason}', "
          f"{len(result.split())} words)")
    cache.set_llm(prompt, model, "openrouter", result)
    return result


# ── Timeline slot builder ─────────────────────────────────────────────────────

def _build_timeline_slots(
    lyrics: list[LyricLine],
    audio_duration: float,
    min_gap: float = _MIN_GAP_FOR_SLOT,
) -> list[dict]:
    """Pre-compute fixed-duration time slots from the lyric timeline.

    Displayable lyrics (start_time < end_time) each get their own slot.
    Silence gaps between lyrics become their own slots if ≥ min_gap seconds.
    An intro silence and/or outro silence slot is added if present.

    Returns a list of dicts:
        {
            "slot_index":    int,
            "start":         float,       # seconds from audio start
            "end":           float,       # seconds from audio start
            "duration":      float,       # end - start
            "lyric_indices": list[int],   # indices into lyrics list (empty for silence)
            "has_lyrics":    bool,
            "label":         str,         # human-readable slot label
        }
    """
    displayable = [(i, ll) for i, ll in enumerate(lyrics) if ll.end_time > ll.start_time]

    if not displayable:
        return [{
            "slot_index": 0, "start": 0.0, "end": audio_duration,
            "duration": audio_duration, "lyric_indices": [], "has_lyrics": False,
            "label": "SILENCE/INSTRUMENTAL",
        }]

    slots = []
    slot_idx = 0
    t = 0.0

    for seq_i, (ly_idx, ll) in enumerate(displayable):
        gap = ll.start_time - t
        if gap >= min_gap:
            label = "INTRO SILENCE" if t == 0.0 else "SILENCE GAP"
            slots.append({
                "slot_index": slot_idx,
                "start": t, "end": ll.start_time,
                "duration": gap,
                "lyric_indices": [], "has_lyrics": False,
                "label": label,
            })
            slot_idx += 1

        lyric_duration = ll.end_time - ll.start_time
        is_long = lyric_duration >= _MULTICLIP_THRESHOLD
        label = f'"{ll.text}"' + (" ← LONG: split into 2–3 clips!" if is_long else "")
        slots.append({
            "slot_index": slot_idx,
            "start": ll.start_time, "end": ll.end_time,
            "duration": lyric_duration,
            "lyric_indices": [ly_idx], "has_lyrics": True,
            "label": label,
        })
        slot_idx += 1
        t = ll.end_time

    # Outro silence
    outro_gap = audio_duration - t
    if outro_gap >= min_gap:
        slots.append({
            "slot_index": slot_idx,
            "start": t, "end": audio_duration,
            "duration": outro_gap,
            "lyric_indices": [], "has_lyrics": False,
            "label": "OUTRO SILENCE",
        })

    return slots


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_plan(
    scenes: list[Scene],
    lyrics: list[LyricLine],
    film_name: str,
    song_name: str,
    output_dir: Path,
    slug: str = "output",
    model: str | None = None,
    characters: list[str] | None = None,
    audio_path: Path | None = None,
) -> list[MatchedSegment]:
    """Ask an LLM via OpenRouter to propose a scene-to-lyric mapping.

    The plan is anchored to the audio timeline: every slot's duration is fixed by
    the lyric timestamps.  The LLM only chooses which scene to use (and where within
    it to start); clip durations are not up for negotiation.

    Args:
        audio_path: Path to the audio file (used to probe total duration).
                    Falls back to the last lyric's end_time if None.

    Saves two files:
      - {output_dir}/{slug}_plan.json   (machine-readable, user can edit)
      - {output_dir}/{slug}_plan_readable.txt  (human-friendly preview)

    Returns the MatchedSegment list so the caller can proceed directly if desired.
    """
    film_scenes = [s for s in scenes if s.is_usable]
    if not film_scenes:
        print("[matcher] Warning: no usable scenes found — using all scenes")
        film_scenes = scenes

    # ── Audio duration ────────────────────────────────────────────────────────
    if audio_path is not None:
        audio_duration = _probe_audio_duration(audio_path)
        if audio_duration <= 0:
            print("[matcher] Warning: ffprobe returned 0 duration, falling back to lyrics")
            audio_duration = 0.0

    if (audio_path is None) or (audio_duration <= 0):
        # Fallback: use last displayable lyric's end_time
        displayable = [ll for ll in lyrics if ll.end_time > ll.start_time]
        audio_duration = displayable[-1].end_time if displayable else 0.0
        print(f"[matcher] Audio duration derived from lyrics: {audio_duration:.2f}s")
    else:
        audio_duration = min(audio_duration, MAX_REEL_DURATION)
        print(f"[matcher] Audio duration (ffprobe): {audio_duration:.2f}s")

    # ── Build timeline slots ──────────────────────────────────────────────────
    slots = _build_timeline_slots(lyrics, audio_duration)
    print(f"[matcher] Timeline: {len(slots)} slots over {audio_duration:.2f}s")

    # ── Call LLM ─────────────────────────────────────────────────────────────
    prompt = _build_prompt(film_scenes, lyrics, film_name, song_name, audio_duration, slots, characters=characters)
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
        # Try to recover a partial array (truncated JSON ends mid-object)
        truncated = raw.rstrip().rstrip(",")
        if not truncated.endswith("]"):
            last_close = truncated.rfind("}")
            if last_close != -1:
                truncated = truncated[: last_close + 1] + "]"
                try:
                    plan_data = json.loads(truncated)
                    print(
                        f"[matcher] ⚠️  JSON was truncated — recovered {len(plan_data)} complete "
                        f"entries. Re-run generate-plan to get the full plan."
                    )
                except json.JSONDecodeError:
                    raise ValueError(
                        f"[matcher] LLM returned invalid/truncated JSON: {e}\nRaw:\n{raw[:1000]}"
                    )
            else:
                raise ValueError(
                    f"[matcher] LLM returned invalid JSON: {e}\nRaw:\n{raw[:1000]}"
                )
        else:
            raise ValueError(
                f"[matcher] LLM returned invalid JSON: {e}\nRaw:\n{raw[:1000]}"
            )

    # ── Parse new slot-based format ───────────────────────────────────────────
    scene_by_index = {s.index: s for s in scenes}
    slot_map = {sl["slot_index"]: sl for sl in slots}

    # Group entries by slot_index, preserving order within each slot
    slot_entries: dict[int, list[dict]] = defaultdict(list)
    for entry in plan_data:
        si = entry.get("slot_index")
        if si is None:
            print("[matcher] Warning: entry missing slot_index, skipping")
            continue
        slot_entries[si].append(entry)

    segments: list[MatchedSegment] = []

    for slot_idx in sorted(slot_entries.keys()):
        if slot_idx not in slot_map:
            print(f"[matcher] Warning: slot_index {slot_idx} not in timeline — skipping")
            continue

        slot = slot_map[slot_idx]
        entries = slot_entries[slot_idx]
        slot_duration = slot["duration"]
        slot_lyrics = [lyrics[i] for i in slot["lyric_indices"] if 0 <= i < len(lyrics)]

        # Scale clip_durations to exactly match slot_duration (handle LLM rounding errors)
        raw_durs = [max(0.01, e.get("clip_duration", slot_duration)) for e in entries]
        total_raw = sum(raw_durs)
        scale = slot_duration / total_raw

        t = slot["start"]
        for entry, raw_dur in zip(entries, raw_durs):
            actual_dur = raw_dur * scale

            scene_idx = entry.get("scene_index")
            if scene_idx not in scene_by_index:
                print(f"[matcher] Warning: entry references unknown scene {scene_idx} — skipping")
                t += actual_dur
                continue

            scene = scene_by_index[scene_idx]
            scene_src_dur = scene.end_time - scene.start_time

            trim_start = float(entry.get("scene_trim_start", 0.0))
            trim_start = max(0.0, min(trim_start, max(0.0, scene_src_dur - actual_dur)))
            trim_end = min(trim_start + actual_dur, scene_src_dur)

            segments.append(
                MatchedSegment(
                    scene_index=scene_idx,
                    lyric_lines=slot_lyrics,
                    scene_trim_start=trim_start,
                    scene_trim_end=trim_end,
                    song_start=t,
                    song_end=t + actual_dur,
                )
            )
            t += actual_dur

    # ── Coverage check ────────────────────────────────────────────────────────
    covered_slots  = set(slot_entries.keys())
    all_slot_idxs  = set(slot_map.keys())
    missing_slots  = all_slot_idxs - covered_slots

    if missing_slots:
        missing_labels = [
            f"[{si}] {slot_map[si]['label']}"
            for si in sorted(missing_slots)
            if slot_map[si]["has_lyrics"]
        ]
        print(
            f"[matcher] ⚠️  {len(missing_slots)} slot(s) not covered in plan "
            f"(indices: {sorted(missing_slots)}).\n"
            f"  Missing lyric slots: {missing_labels}\n"
            f"  → Delete the cached plan and re-run generate-plan to get a complete plan."
        )
    else:
        print(f"[matcher] ✓ All {len(all_slot_idxs)} timeline slots covered.")

    # Duration sanity check
    total_plan_dur = (segments[-1].song_end if segments else 0.0)
    print(f"[matcher] Plan timeline: 0.00s → {total_plan_dur:.2f}s  (audio: {audio_duration:.2f}s)")
    if abs(total_plan_dur - audio_duration) > 0.5:
        print(
            f"[matcher] ⚠️  Plan duration ({total_plan_dur:.2f}s) differs from audio "
            f"({audio_duration:.2f}s) — check for missing slots above."
        )

    # ── Save files ────────────────────────────────────────────────────────────
    plan_path = output_dir / f"{slug}_plan.json"
    with open(plan_path, "w") as f:
        json.dump([s.to_dict() for s in segments], f, indent=2)

    readable_path = output_dir / f"{slug}_plan_readable.txt"
    _write_readable_plan(segments, scene_by_index, slots, film_name, song_name, readable_path)

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


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    scenes: list[Scene],
    lyrics: list[LyricLine],
    film_name: str,
    song_name: str,
    audio_duration: float,
    slots: list[dict],
    characters: list[str] | None = None,
) -> str:
    _STAR  = "★"
    _EMPTY = "☆"

    scene_parts = []
    for s in scenes:
        chars     = ", ".join(s.characters_present) if s.characters_present else "no named characters"
        emotion_s = s.emotion   if s.emotion   else "—"
        shot_s    = s.shot_type if s.shot_type else "—"
        light_s   = s.lighting  if s.lighting  else "—"
        vp        = max(1, min(5, s.visual_power))
        stars     = _STAR * vp + _EMPTY * (5 - vp)
        scene_parts.append(
            f"  Scene {s.index} [{s.start_time:.1f}s–{s.end_time:.1f}s, {s.duration:.1f}s]"
            f"  VisualPower:{stars}({vp}/5)\n"
            f"    Characters : {chars}\n"
            f"    Emotion    : {emotion_s}\n"
            f"    Shot       : {shot_s}\n"
            f"    Lighting   : {light_s}\n"
            f"    Visual     : {s.description}"
        )
    scene_lines = "\n\n".join(scene_parts)

    # Slots display
    slot_lines = []
    for sl in slots:
        dur_str = f"{sl['duration']:.2f}s"
        sym     = "♪" if sl["has_lyrics"] else "⬚"
        slot_lines.append(
            f"  Slot {sl['slot_index']:>2}  "
            f"[{sl['start']:>7.2f}s – {sl['end']:>7.2f}s, {dur_str:>6}]  "
            f"{sym} {sl['label']}"
        )
    slots_display = "\n".join(slot_lines)

    # Characters block
    if characters:
        char_names = ", ".join(characters)
        char_block = f"""
MAIN CHARACTERS TO PRIORITIZE: {char_names}
Strongly prefer scenes featuring these characters — especially close-ups of their faces,
intimate moments between them, and scenes where their emotional state directly mirrors
the lyric. A great fan-edit feels personal, not like a highlight reel of action shots.
"""
    else:
        char_block = ""

    total_slots = len(slots)
    lyric_slots = sum(1 for sl in slots if sl["has_lyrics"])
    silence_slots = total_slots - lyric_slots

    return f"""You are a world-class video editor creating a deeply resonant Instagram Reel from the film '{film_name}' set to the song '{song_name}'.
{char_block}
AVAILABLE SCENES (only aesthetically clean, film-related scenes):
{scene_lines}

══════════════════════════════════════════════════════════════════════════════
AUDIO TIMELINE — {audio_duration:.2f}s total  ({lyric_slots} lyric slots + {silence_slots} silence slots = {total_slots} total)
══════════════════════════════════════════════════════════════════════════════
IMPORTANT: These time slots are IMMUTABLE. Their durations are derived from the
lyrics transcript and cannot be changed. Lyric timestamps mark the EXACT moments
a phrase is pronounced — they are ground truth.

Your only decisions: (1) which scene to show in each slot, (2) where in that scene
to start the clip. The output video MUST cover the full {audio_duration:.2f}s exactly.

{slots_display}

══════════════════════════════════════════════════════════════════════════════

SCENE SELECTION PRINCIPLES (apply in priority order):
1. EMOTIONAL RESONANCE — the scene's emotional register must match the lyric
   • Tender/longing → intimate close-ups, quiet moments, characters apart
   • Euphoric/free  → movement, brightness, characters together
   • Melancholy     → stillness, distance, averted eyes, fading light
   • Tension        → confrontation, charged silence, uncertain expression

2. CHARACTER FOCUS — faces and readable reactions beat any scenery shot.
   A character's expression reacting to a lyric's words is worth 10 landscape shots.

3. VISUAL POWER — prefer higher ★ scenes. Never use ★★☆☆☆ when a ★★★+ alternative
   with matching emotion exists. Silence/gap slots still deserve beautiful visuals.

4. LYRIC-VISUAL SYNC — think literally AND metaphorically:
   • "Waiting" → stillness, empty space, a character poised
   • "Running" → motion, urgency, escape
   • A question → uncertain or searching expression
   • An answer  → resolution, arrival, eye contact

5. VARIETY — avoid repeating the same scene_index on consecutive clips.

TASK:
Assign a scene clip to every slot in the timeline above and return a JSON array.

CRITICAL RULES FOR DURATIONS:
• For each slot: the sum of clip_duration values MUST equal the slot's duration EXACTLY.
• For slots ≤ {_MULTICLIP_THRESHOLD:.0f}s: use ONE clip (clip_duration = slot duration).
• For slots > {_MULTICLIP_THRESHOLD:.0f}s: use 2–3 clips to create visual rhythm
  (split the slot duration across them; they must still add up to the slot duration).
• scene_trim_start + clip_duration must not exceed the scene's source duration.

Each JSON element must have:
  "slot_index":        <integer — must match one of the slot indices above>,
  "scene_index":       <integer — must be one of the available scene indices>,
  "scene_trim_start":  <float — seconds into the scene where the clip starts>,
  "clip_duration":     <float — MUST equal slot duration (single clip) or sum to slot duration (multi-clip)>,
  "reasoning":         "<one sentence: scene emotion + why it fits this slot>"

Return ONLY valid JSON, no markdown, no extra text."""


# ── Human-readable plan writer ────────────────────────────────────────────────

def _write_readable_plan(
    segments: list[MatchedSegment],
    scene_by_index: dict[int, Scene],
    slots: list[dict],
    film_name: str,
    song_name: str,
    out_path: Path,
) -> None:
    lines = [
        f"PLAN: '{film_name}' × '{song_name}'",
        "=" * 60,
        "",
    ]

    # Build a slot_index → segment mapping for the summary header
    slot_map = {sl["slot_index"]: sl for sl in slots}

    for i, seg in enumerate(segments, 1):
        scene = scene_by_index.get(seg.scene_index)
        scene_desc  = scene.description if scene else "unknown"
        scene_start = scene.start_time  if scene else 0
        scene_end   = scene.end_time    if scene else 0

        trim_start  = seg.scene_trim_start
        trim_end    = seg.scene_trim_end
        clip_dur    = trim_end - trim_start if trim_end >= 0 else (scene_end - scene_start - trim_start)

        vp    = scene.visual_power if scene else 3
        stars = "★" * vp + "☆" * (5 - vp)
        chars = ", ".join(scene.characters_present) if scene and scene.characters_present else "—"

        # Find which slot this segment belongs to (by song_start)
        slot_label = ""
        if seg.song_start >= 0:
            for sl in slots:
                if abs(sl["start"] - seg.song_start) < 0.05:
                    slot_label = f"  Slot {sl['slot_index']}  {sl['label']}"
                    break

        song_pos = (
            f"[{seg.song_start:.2f}s–{seg.song_end:.2f}s in audio]"
            if seg.song_start >= 0 else "[legacy: no song_start]"
        )

        lines.append(f"Clip {i}  {song_pos}  (dur: {clip_dur:.2f}s){slot_label}")
        lines.append(f"  Scene {seg.scene_index} [{scene_start:.1f}s–{scene_end:.1f}s in source]  Power:{stars}({vp}/5)")
        lines.append(f"  Characters : {chars}")
        lines.append(f"  Emotion    : {scene.emotion if scene else '—'}")
        lines.append(f"  Shot       : {scene.shot_type if scene else '—'}  |  Lighting: {scene.lighting if scene else '—'}")
        lines.append(f"  Trim: {trim_start:.2f}s → {trim_end:.2f}s within scene")
        lines.append(f"  Visual: {scene_desc}")
        if seg.lyric_lines:
            lines.append(f"  Lyrics:")
            for lyric in seg.lyric_lines:
                lines.append(f"    [{lyric.start_time:.2f}s–{lyric.end_time:.2f}s] {lyric.text}")
        lines.append("")

    total_dur = segments[-1].song_end if segments else 0.0
    lines.append(f"Total duration: {total_dur:.2f}s")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
