"""Match scenes to lyric lines using an LLM via OpenRouter. Produces a reviewable plan file.

The plan is built around the FIXED lyric timeline — every slot's duration is immutable.
The LLM's only job is to select which scene (and where within it) to show in each slot.
"""
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
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

# Lyric slots longer than this get flagged as multi-clip candidates in the prompt.
# Raised to 8s so short lyric lines (2-5s) never get split into micro-clips.
_MULTICLIP_THRESHOLD = 8.0

# Clips shorter than this (seconds) are flagged as problematic after plan generation.
_MIN_CLIP_DURATION = 1.5


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

def _call_openrouter(prompt: str, model: str, cache_dir: Path | None = None) -> str:
    cached = cache.get_llm(prompt, model, "openrouter", cache_dir=cache_dir)
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
    cache.set_llm(prompt, model, "openrouter", result, cache_dir=cache_dir)
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
            # Large enough gap → dedicated silence slot ending exactly at lyric start
            label = "INTRO SILENCE" if t == 0.0 else "SILENCE GAP"
            slots.append({
                "slot_index": slot_idx,
                "start": t, "end": ll.start_time,
                "duration": gap,
                "lyric_indices": [], "has_lyrics": False,
                "label": label,
            })
            slot_idx += 1
            t = ll.start_time   # advance cursor to lyric start
        # Micro-gap (0 < gap < min_gap) or no gap: lyric slot starts at cursor t.
        # This snaps the slot start to the previous slot's end, keeping the chain
        # perfectly gapless. The subtitle timing still uses lyric.start_time from
        # lyrics.json (not the slot boundary), so the difference is imperceptible.

        slot_start = t
        lyric_duration = ll.end_time - slot_start
        is_long = lyric_duration >= _MULTICLIP_THRESHOLD
        label = f'"{ll.text}"' + (" ← LONG: split into 2–3 clips!" if is_long else "")
        slots.append({
            "slot_index": slot_idx,
            "start": slot_start, "end": ll.end_time,
            "duration": max(lyric_duration, 0.01),
            "lyric_indices": [ly_idx], "has_lyrics": True,
            "label": label,
        })
        slot_idx += 1
        t = ll.end_time

    # Outro silence — always add even if tiny, so the plan covers 100% of the audio.
    outro_gap = audio_duration - t
    if outro_gap > 0.01:
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
    cache_dir: Path | None = None,
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
    raw = _call_openrouter(prompt, resolved_model, cache_dir=cache_dir)

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

            raw_trim_start = float(entry.get("scene_trim_start", 0.0))
            trim_start = max(0.0, min(raw_trim_start, max(0.0, scene_src_dur - actual_dur)))
            if abs(trim_start - raw_trim_start) > 0.01:
                print(
                    f"[matcher] ⚠  Slot {slot_idx} scene {scene_idx}: "
                    f"trim_start clamped {raw_trim_start:.3f}s → {trim_start:.3f}s "
                    f"(scene dur {scene_src_dur:.3f}s, need {actual_dur:.3f}s)"
                )

            raw_trim_end = trim_start + actual_dur
            trim_end = min(raw_trim_end, scene_src_dur)
            if abs(trim_end - raw_trim_end) > 0.01:
                print(
                    f"[matcher] ⚠  Slot {slot_idx} scene {scene_idx}: "
                    f"trim_end clamped {raw_trim_end:.3f}s → {trim_end:.3f}s "
                    f"— scene is only {scene_src_dur:.3f}s but clip needs {actual_dur:.3f}s. "
                    f"Pick a longer scene next time."
                )

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

    # ── Coverage check + auto-fill silence gaps ───────────────────────────────
    covered_slots  = set(slot_entries.keys())
    all_slot_idxs  = set(slot_map.keys())
    missing_slots  = all_slot_idxs - covered_slots

    missing_lyric_slots   = [si for si in sorted(missing_slots) if slot_map[si]["has_lyrics"]]
    missing_silence_slots = [si for si in sorted(missing_slots) if not slot_map[si]["has_lyrics"]]

    if missing_lyric_slots:
        labels = [f"[{si}] {slot_map[si]['label']}" for si in missing_lyric_slots]
        print(
            f"[matcher] ⚠️  {len(missing_lyric_slots)} lyric slot(s) not covered: {labels}\n"
            f"  → Re-run generate-plan to get a complete plan."
        )
    if not missing_slots:
        print(f"[matcher] ✓ All {len(all_slot_idxs)} timeline slots covered.")

    # Auto-fill any uncovered SILENCE slots (intro/outro/gap) with the best
    # available scene.  This is the root cause of the "10-second extension" bug:
    # the LLM frequently skips the outro-silence slot, causing render_video() to
    # stretch the last clip by the missing duration and show wrong content.
    if missing_silence_slots:
        used_indices = {seg.scene_index for seg in segments}
        # Prefer unused scenes, sorted by visual_power desc; fall back to reuse if needed
        pool = sorted(
            [s for s in film_scenes if s.index not in used_indices],
            key=lambda s: -s.visual_power,
        ) + sorted(film_scenes, key=lambda s: -s.visual_power)

        fill_segs: list[MatchedSegment] = []
        pool_idx = 0
        for si in missing_silence_slots:
            slot = slot_map[si]
            scene = pool[pool_idx % len(pool)]
            pool_idx += 1
            scene_src_dur = scene.end_time - scene.start_time
            trim_end = min(slot["duration"], scene_src_dur)
            fill_segs.append(MatchedSegment(
                scene_index=scene.index,
                lyric_lines=[],
                scene_trim_start=0.0,
                scene_trim_end=trim_end,
                song_start=slot["start"],
                song_end=slot["end"],
            ))
            print(
                f"[matcher] Auto-filled {slot['label']} slot [{si}] "
                f"({slot['duration']:.1f}s) with scene {scene.index}"
            )

        segments = sorted(segments + fill_segs, key=lambda s: s.song_start)
        print(f"[matcher] ✓ All silence slots filled — total clips: {len(segments)}")

    # Duration sanity check
    total_plan_dur = (segments[-1].song_end if segments else 0.0)
    print(f"[matcher] Plan timeline: 0.00s → {total_plan_dur:.2f}s  (audio: {audio_duration:.2f}s)")
    if abs(total_plan_dur - audio_duration) > 0.5:
        print(
            f"[matcher] ⚠️  Plan duration ({total_plan_dur:.2f}s) differs from audio "
            f"({audio_duration:.2f}s) — check for missing slots above."
        )

    # ── Scene uniqueness check ─────────────────────────────────────────────────
    scene_usage: dict[int, int] = {}
    for seg in segments:
        scene_usage[seg.scene_index] = scene_usage.get(seg.scene_index, 0) + 1
    duplicates = {idx: cnt for idx, cnt in scene_usage.items() if cnt > 1}
    if duplicates:
        top = sorted(duplicates.items(), key=lambda x: -x[1])[:8]
        top_str = ", ".join(f"Scene {idx} ×{cnt}" for idx, cnt in top)
        print(
            f"[matcher] ⚠️  Duplicate scenes in plan: {top_str}\n"
            f"  {len(duplicates)} scene(s) reused — re-run generate-plan (prompt now enforces "
            f"uniqueness so cache will miss) or use edit-plan to swap duplicates."
        )
    else:
        print(
            f"[matcher] ✓ Scene uniqueness OK — {len(scene_usage)} unique scenes "
            f"across {len(segments)} clips."
        )

    # ── Micro-clip check ───────────────────────────────────────────────────────
    micro_clips = [
        seg for seg in segments
        if (seg.song_end - seg.song_start) < _MIN_CLIP_DURATION
    ]
    if micro_clips:
        times = ", ".join(f"{s.song_start:.1f}s" for s in micro_clips[:6])
        print(
            f"[matcher] ⚠️  {len(micro_clips)} clip(s) under {_MIN_CLIP_DURATION}s "
            f"(at: {times}) — may look choppy. "
            f"Use edit-plan to merge or extend them."
        )

    # ── Enforce sequential song_start/song_end invariant ─────────────────────
    # Sort by song_start to fix any out-of-order segments produced by the LLM.
    # Then rebuild a perfectly gapless timeline: first clip starts at 0.0,
    # and every subsequent clip's song_start = previous clip's song_end.
    # Each clip's DURATION is preserved exactly (scene_trim_end - scene_trim_start).
    segments.sort(key=lambda s: s.song_start)

    cursor = 0.0
    rebuilt: list[MatchedSegment] = []
    corrected = 0
    for seg in segments:
        dur = (
            seg.scene_trim_end - seg.scene_trim_start
            if seg.scene_trim_end >= 0
            else seg.song_end - seg.song_start
        )
        dur = max(dur, 0.01)
        if abs(seg.song_start - cursor) > 0.001:
            corrected += 1
            if abs(seg.song_start - cursor) > 0.2:
                print(
                    f"[matcher] song_start corrected for scene {seg.scene_index}: "
                    f"{seg.song_start:.4f}s → {cursor:.4f}s "
                    f"(was {seg.song_start - cursor:+.4f}s off)"
                )
        rebuilt.append(MatchedSegment(
            scene_index=seg.scene_index,
            lyric_lines=seg.lyric_lines,
            scene_trim_start=seg.scene_trim_start,
            scene_trim_end=seg.scene_trim_end,
            song_start=cursor,
            song_end=cursor + dur,
        ))
        cursor += dur

    if corrected:
        print(
            f"[matcher] Sequential fix: {corrected} song_start/song_end adjusted "
            f"→ timeline is now gapless 0.0–{cursor:.4f}s"
        )
        segments = rebuilt

    # ── Archive existing plan before overwriting ──────────────────────────────
    archived = archive_existing_plan(output_dir, slug)
    if archived:
        print(f"[matcher] Archived previous plan → plan_history/{archived.name}")

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


def archive_existing_plan(output_dir: Path, slug: str) -> Path | None:
    """Copy the current plan + readable to plan_history/ before overwriting.

    Returns the archive path, or None if there was no existing plan to archive.
    """
    plan_path = output_dir / f"{slug}_plan.json"
    if not plan_path.exists():
        return None

    history_dir = output_dir / "plan_history"
    history_dir.mkdir(exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_json = history_dir / f"{slug}_plan_{stamp}.json"
    # Avoid same-second collisions (e.g. rapid automated re-runs)
    if archive_json.exists():
        stamp += f"_{datetime.now(timezone.utc).strftime('%f')[:3]}"
        archive_json = history_dir / f"{slug}_plan_{stamp}.json"
    archive_json.write_bytes(plan_path.read_bytes())

    readable = output_dir / f"{slug}_plan_readable.txt"
    if readable.exists():
        (history_dir / f"{slug}_plan_{stamp}_readable.txt").write_bytes(readable.read_bytes())

    return archive_json


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
            # dur= and MAX_TRIM_END= are the binding hard limits the LLM must respect
            f"  Scene {s.index}  dur={s.duration:.2f}s  MAX_TRIM_END={s.duration:.2f}s"
            f"  VisualPower:{stars}({vp}/5)\n"
            f"    Characters : {chars}\n"
            f"    Emotion    : {emotion_s}\n"
            f"    Shot       : {shot_s}\n"
            f"    Lighting   : {light_s}\n"
            f"    Visual     : {s.description}"
        )
    scene_lines = "\n\n".join(scene_parts)

    # Slots display — each slot carries an eligibility list so the LLM never
    # selects a scene that is physically too short to fill the required duration.
    slot_lines = []
    for sl in slots:
        dur_str  = f"{sl['duration']:.2f}s"
        sym      = "♪" if sl["has_lyrics"] else "⬚"
        slot_dur = sl["duration"]

        # Scenes whose source duration is long enough to fill this slot
        eligible = [s for s in scenes if s.duration >= slot_dur]
        n_total  = len(scenes)
        if slot_dur <= 0 or len(eligible) == n_total:
            elig_str = f"all {n_total} scenes eligible"
        elif len(eligible) == 0:
            longest = sorted(scenes, key=lambda s: -s.duration)[:6]
            elig_str = (
                f"⚠ NO scene long enough for {slot_dur:.2f}s! "
                f"Longest available: "
                + ", ".join(f"Sc{s.index}({s.duration:.2f}s)" for s in longest)
            )
        elif len(eligible) <= 20:
            elig_str = "eligible: " + " ".join(str(s.index) for s in eligible)
        else:
            elig_str = f"{len(eligible)}/{n_total} scenes eligible"

        slot_lines.append(
            f"  Slot {sl['slot_index']:>2}  "
            f"[{sl['start']:>7.2f}s – {sl['end']:>7.2f}s, {dur_str:>6}]  "
            f"{sym} {sl['label']}\n"
            f"           need ≥{slot_dur:.2f}s  →  {elig_str}"
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

    total_slots   = len(slots)
    lyric_slots   = sum(1 for sl in slots if sl["has_lyrics"])
    silence_slots = total_slots - lyric_slots
    n_scenes      = len(scenes)
    # Target: roughly one clip per slot; allow a small overage for long slots
    clip_budget   = total_slots + max(4, total_slots // 5)

    return f"""You are a world-class video editor creating a deeply resonant Instagram Reel from the film '{film_name}' set to the song '{song_name}'.
{char_block}
AVAILABLE SCENES ({n_scenes} scenes — ranked by visual_power ★; use ★ as your primary quality filter):
{scene_lines}

══════════════════════════════════════════════════════════════════════════════
AUDIO TIMELINE — {audio_duration:.2f}s total  ({lyric_slots} lyric slots + {silence_slots} silence slots = {total_slots} total)
══════════════════════════════════════════════════════════════════════════════
These time slots are IMMUTABLE. Their durations come from the lyrics transcript
and cannot be changed. Lyric timestamps are ground truth.

Your only decisions: (1) which scene fills each slot, (2) where in that scene
to start the clip.

FULL COVERAGE IS MANDATORY — the output video must fill the song from 0.00s to
{audio_duration:.2f}s with zero gaps. This means:
• The very first slot (intro silence or first lyric) must be assigned a scene.
• Every silence/gap slot between lyrics must be assigned a scene.
• The very last slot (outro silence or last lyric) must be assigned a scene that
  reaches exactly {audio_duration:.2f}s. Do not leave the end of the song uncut.
• Skipping ANY slot — especially intro/outro silence — is an error.

{slots_display}

══════════════════════════════════════════════════════════════════════════════
HARD CONSTRAINTS — non-negotiable, checked automatically after your response:

1. NO SCENE REUSE — each scene_index may appear AT MOST ONCE in the entire plan.
   You have {n_scenes} distinct scenes for {total_slots} slots. There is no excuse to
   repeat a scene. As you assign each slot, mentally cross off that scene_index and
   never select it again.

2. NO MICRO-CLIPS — every individual clip must be ≥ {_MIN_CLIP_DURATION}s.
   Cuts faster than {_MIN_CLIP_DURATION}s look amateur on mobile. If a slot is short,
   fill it with ONE clip — never split it.

3. CLIP BUDGET — target ≤ {clip_budget} clips total (≈ one per slot).
   Multi-clip splits are only justified for slots longer than {_MULTICLIP_THRESHOLD:.0f}s.
   Do not pad the edit with unnecessary cuts.
══════════════════════════════════════════════════════════════════════════════

SCENE SELECTION PRINCIPLES (apply after satisfying the hard constraints above):

1. VISUAL POWER FIRST — ★ is the primary quality signal. Prefer ★★★★+ scenes.
   Never use ★★☆☆☆ when a ★★★+ alternative with matching emotion exists.
   Silence/gap slots (intro, outro, gaps) deserve visually strong scenes just as
   much as lyric slots — they are the emotional framing of the whole piece.

2. EMOTIONAL RESONANCE — match the scene's register to the lyric or mood
   • Tender/longing → intimate close-ups, quiet moments, characters apart
   • Euphoric/free  → movement, brightness, characters together
   • Melancholy     → stillness, distance, averted eyes, fading light
   • Tension        → confrontation, charged silence, uncertain expression

3. CHARACTER FOCUS — faces and readable reactions beat any scenery shot.
   A character's expression reacting to a lyric's words is worth 10 landscape shots.

4. LYRIC-VISUAL SYNC — think literally AND metaphorically:
   • "Waiting" → stillness, empty space, a character poised
   • "Running" → motion, urgency, escape
   • A question → uncertain or searching expression
   • An answer  → resolution, arrival, eye contact

5. SHOT VARIETY — vary shot scale on consecutive clips (close-up → medium → wide → …)
   to create visual rhythm even within the no-reuse constraint.

TASK:
Assign a scene clip to EVERY slot in the timeline above (all {total_slots} slots, no exceptions)
and return a JSON array.

CLIP DURATION RULES:
• For each slot: the sum of clip_duration values MUST equal the slot's duration EXACTLY.
• Slots ≤ {_MULTICLIP_THRESHOLD:.0f}s → exactly ONE clip (clip_duration = slot duration).
• Slots > {_MULTICLIP_THRESHOLD:.0f}s → 2–3 clips for visual rhythm, each ≥ {_MIN_CLIP_DURATION}s,
  summing to the exact slot duration.

HARD TRIM CONSTRAINT — enforced mathematically after your response:
• Every scene shows its MAX_TRIM_END in the scene list above. That is the hard ceiling.
• scene_trim_start ≥ 0 always.
• scene_trim_end = scene_trim_start + clip_duration — this computed value MUST be ≤ MAX_TRIM_END.
  Concretely: scene_trim_start ≤ MAX_TRIM_END − clip_duration.
  Example: scene dur=3.50s, clip_duration=2.80s → scene_trim_start ≤ 0.70s.
• ONLY use scenes listed under "eligible" for each slot. If a scene's dur < clip_duration
  it will be silently truncated, producing a frozen last frame in the output video.
• Output ONLY scene_trim_start. Do NOT output scene_trim_end — it is computed automatically.

Each JSON element must have:
  "slot_index":        <integer — must match one of the slot indices above>,
  "scene_index":       <integer — must be in the slot's eligible list>,
  "scene_trim_start":  <float — seconds into the scene; MUST satisfy: scene_trim_start + clip_duration ≤ scene.MAX_TRIM_END>,
  "clip_duration":     <float — MUST equal slot duration (single clip) or sum to it (multi-clip)>,
  "reasoning":         "<one sentence: scene emotion + why it fits this slot>"

OUTPUT ORDER AND CONTIGUITY:
• Output entries sorted by slot_index (ascending) — never skip or re-order slots.
• Every slot in the timeline above MUST have at least one entry.
• Taken in order, the entries must form a gapless chain: the first entry's slot
  starts at {slots[0]["start"]:.3f}s and the last entry's slot ends at {slots[-1]["end"]:.3f}s.
• A missing slot leaves a hole in the video timeline and corrupts sync.

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
