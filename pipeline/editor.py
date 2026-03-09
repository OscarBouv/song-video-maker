"""Assemble the final 9:16 Instagram Reel using a single ffmpeg pass.

Portrait conversion uses blur-background:
  - Background: source video scaled to fill 1080×1920, center-cropped, Gaussian-blurred.
  - Foreground: source video scaled to *fit* inside 1080×1920 (letterboxed, no crop),
    overlaid centred on the blurred background.
This means no content is ever hard-cropped out — the blur fills the black bars.

Subtitles are rendered via ffmpeg's native drawtext filter for best sharpness.
"""
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from config import (
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    OUTPUT_FPS,
    MAX_REEL_DURATION,
    SUBTITLE_COLOR,
    SUBTITLE_FONTSIZE,
    SUBTITLE_Y_RATIO,
    SUBTITLE_FONT,
    FFMPEG_BIN,
    FFPROBE_BIN,
)
from models import Scene, MatchedSegment


def render_video(
    segments: list[MatchedSegment],
    scenes: list[Scene],
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    max_duration: float = MAX_REEL_DURATION,
) -> None:
    """Single-pass ffmpeg render: trim clips → blur-background 9:16 crop → drawtext subtitles → audio."""

    video_duration = _probe_duration(video_path)
    scene_by_index = {s.index: s for s in scenes}

    # Detect and remove any hardcoded letterbox / pillarbox bars from the source
    source_crop = _detect_source_crop(video_path)

    # ── Build clip plan ───────────────────────────────────────────────────────
    # Each entry: (src_start, src_end, [(lyric_text, abs_t_start, abs_t_end), ...])
    clips: list[tuple[float, float, list[tuple[str, float, float]]]] = []
    running_time = 0.0

    for seg in segments:
        scene = scene_by_index.get(seg.scene_index)
        if scene is None:
            print(f"[editor] Warning: scene {seg.scene_index} not found, skipping")
            continue

        clip_start = scene.start_time + seg.scene_trim_start
        clip_end = (
            scene.end_time if seg.scene_trim_end < 0
            else scene.start_time + seg.scene_trim_end
        )
        clip_end = min(clip_end, video_duration)
        clip_dur  = clip_end - clip_start

        if clip_dur <= 0:
            print(f"[editor] Warning: scene {seg.scene_index} has zero duration, skipping")
            continue

        if running_time + clip_dur > max_duration:
            clip_dur  = max_duration - running_time
            clip_end  = clip_start + clip_dur
            if clip_dur <= 0:
                break

        # Map each lyric's pronunciation window to absolute positions in the output timeline.
        #
        # NEW format (song_start ≥ 0): lyric.start_time / end_time ARE the absolute output
        # positions (they live on the same audio timeline as the assembled video).  We simply
        # clamp them to this clip's window so the same lyric can safely appear in multiple
        # sub-clips (multi-clip long-lyric slots) without bleeding outside the clip.
        #
        # LEGACY format (song_start = -1): fall back to the old running_time + relative-offset
        # arithmetic (works for old plan files without song_start/song_end).
        subtitles: list[tuple[str, float, float]] = []
        if seg.song_start >= 0:
            clip_song_start = seg.song_start
            clip_song_end   = seg.song_end
            for lyric in seg.lyric_lines:
                # Exact pronunciation window, clamped to this clip's song-time window
                t0 = max(clip_song_start, lyric.start_time)
                t1 = min(clip_song_end,   lyric.end_time)
                if t1 <= t0:
                    continue  # zero-duration marker or entirely outside this clip → skip
                subtitles.append((lyric.text, t0, t1))
        else:
            # Legacy: infer song-start from the first lyric, then compute running_time offsets
            seg_t0 = seg.lyric_lines[0].start_time if seg.lyric_lines else 0.0
            for lyric in seg.lyric_lines:
                t_start = max(0.0, lyric.start_time - seg_t0)
                t_end   = min(clip_dur, lyric.end_time - seg_t0)
                if t_end <= t_start:
                    continue  # zero-duration or inverted → never display, skip
                subtitles.append((lyric.text, running_time + t_start, running_time + t_end))

        clips.append((clip_start, clip_end, subtitles))
        running_time += clip_dur

        if running_time >= max_duration:
            break

    if not clips:
        raise RuntimeError("[editor] No clips to render — check your plan and scenes")

    # ── Extend last clip to fill the full audio ───────────────────────────────
    # Plans may end a fraction of a second short of the audio file's actual
    # duration (floating-point rounding, LLM timing approximations, etc.).
    # Rather than cutting the song mid-fade, we simply stretch the last clip's
    # source end-point to cover the tail — the same scene keeps playing while
    # the audio finishes naturally.
    audio_duration = _probe_duration(audio_path)
    audio_end      = min(audio_duration, max_duration)

    if audio_end > running_time and clips:
        gap = audio_end - running_time
        last_start, last_end, last_subs = clips[-1]
        extended_end = min(last_end + gap, video_duration)
        added = extended_end - last_end
        if added > 0:
            clips[-1] = (last_start, extended_end, last_subs)
            print(f"[editor] Extended last clip by {added:.3f}s to reach audio end ({audio_end:.2f}s)")

    filter_complex = _build_filter_complex(clips, source_crop=source_crop)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[editor] Rendering {len(clips)} clips ({running_time:.1f}s total) → {output_path}...")

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "1:a",
        "-t", str(audio_end),
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-r", str(OUTPUT_FPS),
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"[editor] ffmpeg failed (exit code {result.returncode})")

    print(f"[editor] Done! Output: {output_path}")


# ── Filter-complex builder ────────────────────────────────────────────────────

def _build_filter_complex(
    clips: list[tuple[float, float, list[tuple[str, float, float]]]],
    source_crop: tuple[int, int, int, int] | None = None,
) -> str:
    parts: list[str] = []
    n = len(clips)

    # 1. Trim each clip from the source video
    for i, (start, end, _) in enumerate(clips):
        parts.append(
            f"[0:v]trim=start={start:.6f}:end={end:.6f},"
            f"setpts=PTS-STARTPTS[c{i}]"
        )

    # 2. Concatenate (or pass-through for single clip)
    if n == 1:
        parts.append("[c0]null[vcat]")
    else:
        concat_in = "".join(f"[c{i}]" for i in range(n))
        parts.append(f"{concat_in}concat=n={n}:v=1:a=0[vcat]")

    # 3. Blur-background portrait conversion
    #    bg: scale to cover 9:16 → center-crop → Gaussian blur (kept full-frame, bars blur away)
    #    fg: optionally crop away hardcoded letterbox/pillarbox bars, then scale to fit inside 9:16
    parts.append("[vcat]split=2[bgraw][fgraw]")
    parts.append(
        f"[bgraw]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
        f":force_original_aspect_ratio=increase,"
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},"
        f"gblur=sigma=30[bg]"
    )
    # Apply source crop to foreground only (removes encoded black bars)
    if source_crop:
        cw, ch, cx, cy = source_crop
        parts.append(
            f"[fgraw]crop={cw}:{ch}:{cx}:{cy},"
            f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
            f":force_original_aspect_ratio=decrease:flags=lanczos[fg]"
        )
    else:
        parts.append(
            f"[fgraw]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
            f":force_original_aspect_ratio=decrease:flags=lanczos[fg]"
        )
    parts.append("[bg][fg]overlay=x=(W-w)/2:y=(H-h)/2[vbg]")

    # 4. Subtitle drawtext (chained filters, one per lyric line)
    all_subs: list[tuple[str, float, float]] = [
        (text, t0, t1)
        for (_, _, subs) in clips
        for (text, t0, t1) in subs
    ]

    if all_subs:
        dt_parts = []
        for text, t0, t1 in all_subs:
            safe = _escape_drawtext(text)
            dt_parts.append(
                f"drawtext="
                f"fontfile='{SUBTITLE_FONT}':"
                f"text='{safe}':"
                f"x=(w-tw)/2:"
                f"y=h*{SUBTITLE_Y_RATIO}:"
                f"fontsize={SUBTITLE_FONTSIZE}:"
                f"fontcolor={SUBTITLE_COLOR}@0.92:"
                f"shadowx=2:shadowy=2:shadowcolor=black@0.65:"
                f"enable='between(t\\,{t0:.3f}\\,{t1:.3f})'"
            )
        parts.append(f"[vbg]{','.join(dt_parts)}[vout]")
    else:
        parts.append("[vbg]copy[vout]")

    return ";".join(parts)


def _escape_drawtext(text: str) -> str:
    """Escape special characters for ffmpeg drawtext text and fontfile options."""
    return (
        text
        .replace("\\", "\\\\")   # backslash must come first
        .replace("'",  "\u2019") # straight quote → curly (avoids option quoting issues)
        .replace(":",  "\\:")    # colon separates filter options
        .replace(",",  "\\,")    # comma separates chained filters
        .replace("[",  "\\[")    # brackets used in filter-graph syntax
        .replace("]",  "\\]")
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_source_crop(video_path: Path) -> tuple[int, int, int, int] | None:
    """Scan the source video with cropdetect to find any hardcoded black bars.

    Samples up to the first 90 seconds, collects all detected crop rectangles,
    picks the most-common result (mode), and returns (w, h, x, y) only when the
    crop is meaningfully smaller than the original frame (≥ 1 % reduction in either
    dimension).  Returns None when no significant bars are detected so the
    filter-complex can skip the crop step entirely.
    """
    # ── Get native dimensions ─────────────────────────────────────────────────
    info = subprocess.run(
        [
            FFPROBE_BIN, "-v", "quiet",
            "-print_format", "json",
            "-show_entries", "stream=width,height",
            "-select_streams", "v:0",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    try:
        streams = json.loads(info.stdout).get("streams", [])
        orig_w = int(streams[0]["width"])
        orig_h = int(streams[0]["height"])
    except (KeyError, IndexError, json.JSONDecodeError):
        print("[editor] cropdetect: could not read video dimensions — skipping")
        return None

    # ── Run cropdetect (limit to 90 s so it finishes quickly) ────────────────
    result = subprocess.run(
        [
            FFMPEG_BIN, "-i", str(video_path),
            "-vf", "cropdetect=limit=24:round=2:skip=0",
            "-t", "90",
            "-f", "null", "/dev/null",
        ],
        capture_output=True, text=True,
    )

    # Parse every "crop=W:H:X:Y" occurrence from stderr
    matches = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", result.stderr)
    if not matches:
        print("[editor] cropdetect: no crop values found — skipping")
        return None

    # Use the modal (most-common) crop rectangle for robustness
    (cw_s, ch_s, cx_s, cy_s) = Counter(matches).most_common(1)[0][0]
    cw, ch, cx, cy = int(cw_s), int(ch_s), int(cx_s), int(cy_s)

    # Skip when the crop is essentially the full frame (< 1 % reduction)
    if cw >= orig_w * 0.99 and ch >= orig_h * 0.99:
        print(f"[editor] cropdetect: no significant bars ({orig_w}×{orig_h} → {cw}×{ch}) — skipping")
        return None

    print(
        f"[editor] cropdetect: {orig_w}×{orig_h} → {cw}×{ch} at ({cx},{cy}) "
        f"— removing {orig_h - ch}px vertical / {orig_w - cw}px horizontal bars"
    )
    return cw, ch, cx, cy


def _probe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            FFPROBE_BIN, "-v", "quiet",
            "-print_format", "json",
            "-show_entries", "format=duration",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def _segment_song_start(seg: MatchedSegment) -> float:
    """Return the song timestamp at which this segment starts.

    Prefers the explicit song_start field (set by the timeline-based matcher).
    Falls back to the first lyric's start_time for legacy plan files.
    """
    if seg.song_start >= 0:
        return seg.song_start
    return seg.lyric_lines[0].start_time if seg.lyric_lines else 0.0
