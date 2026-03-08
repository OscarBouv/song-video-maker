"""Assemble the final 9:16 Instagram Reel using a single ffmpeg pass.

Portrait conversion uses blur-background:
  - Background: source video scaled to fill 1080×1920, center-cropped, Gaussian-blurred.
  - Foreground: source video scaled to *fit* inside 1080×1920 (letterboxed, no crop),
    overlaid centred on the blurred background.
This means no content is ever hard-cropped out — the blur fills the black bars.

Subtitles are rendered via ffmpeg's native drawtext filter for best sharpness.
"""
import json
import subprocess
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

        # Convert lyric song-timestamps → absolute positions in the output timeline
        seg_t0 = _segment_song_start(seg)
        subtitles: list[tuple[str, float, float]] = []
        for lyric in seg.lyric_lines:
            t_start = max(0.0, lyric.start_time - seg_t0)
            t_end   = min(clip_dur, lyric.end_time - seg_t0)
            if t_end <= t_start:
                t_start, t_end = 0.0, clip_dur
            subtitles.append((lyric.text, running_time + t_start, running_time + t_end))

        clips.append((clip_start, clip_end, subtitles))
        running_time += clip_dur

        if running_time >= max_duration:
            break

    if not clips:
        raise RuntimeError("[editor] No clips to render — check your plan and scenes")

    # ── Single ffmpeg call ────────────────────────────────────────────────────
    audio_end      = min(running_time, _probe_duration(audio_path))
    filter_complex = _build_filter_complex(clips)

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
    clips: list[tuple[float, float, list[tuple[str, float, float]]]]
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
    #    bg: scale to cover 9:16 → center-crop → Gaussian blur
    #    fg: scale to fit inside 9:16 (Lanczos, no crop) → overlay centred on bg
    parts.append("[vcat]split=2[bgraw][fgraw]")
    parts.append(
        f"[bgraw]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
        f":force_original_aspect_ratio=increase,"
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},"
        f"gblur=sigma=30[bg]"
    )
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
    """Return the song timestamp at which this segment starts (first lyric's start_time)."""
    return seg.lyric_lines[0].start_time if seg.lyric_lines else 0.0
