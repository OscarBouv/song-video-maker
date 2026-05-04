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
    SUBTITLE_BORDER_COLOR,
    SUBTITLE_BORDER_WIDTH,
    SUBTITLE_FONTSIZE,
    SUBTITLE_Y_RATIO,
    SUBTITLE_FONT,
    INSERT_FONT,
    INSERT_FONTSIZE,
    INSERT_COLOR,
    INSERT_BORDER_WIDTH,
    INSERT_X_RATIO,
    INSERT_Y_TOP_RATIO,
    INSERT_START_T,
    INSERT_FADE_T,
    INSERT_END_T,
    INSERT_FADE_OUT_T,
    VIDEO_FADE_DURATION,
    AUDIO_FADE_DURATION,
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
    artist: str = "",
    song_title: str = "",
    film_name: str = "",
    director: str = "",
    subtitle_font: str = "",   # absolute path; empty = use SUBTITLE_FONT from config
    subtitle_fontsize: int = 0,  # 0 = use config default
    subtitle_italic: bool = False,
    manual_crop: dict | None = None,  # {top, bottom, left, right} px — overrides auto-detect
    lyrics=None,               # list[LyricLine] from lyrics.json — bypasses plan lyric_lines
) -> None:
    """Single-pass ffmpeg render: trim clips → blur-background 9:16 crop → drawtext subtitles → audio."""

    # Sort by song_start so any out-of-order plan produces a coherent video.
    # Plans saved after the sequential-fix post-processing are already sorted;
    # this guards against older plans loaded from disk.
    segments = sorted(segments, key=lambda s: s.song_start)

    video_duration = _probe_duration(video_path)
    scene_by_index = {s.index: s for s in scenes}

    # Crop: use manual settings if provided, otherwise auto-detect
    if manual_crop and any(manual_crop.get(k, 0) for k in ("top", "bottom", "left", "right")):
        info = subprocess.run(
            [FFPROBE_BIN, "-v", "quiet", "-print_format", "json",
             "-show_entries", "stream=width,height", "-select_streams", "v:0", str(video_path)],
            capture_output=True, text=True,
        )
        try:
            streams = json.loads(info.stdout).get("streams", [])
            vw, vh = int(streams[0]["width"]), int(streams[0]["height"])
        except Exception:
            vw, vh = 1920, 1080
        top    = manual_crop.get("top", 0)
        bottom = manual_crop.get("bottom", 0)
        left   = manual_crop.get("left", 0)
        right  = manual_crop.get("right", 0)
        cw, ch, cx, cy = vw - left - right, vh - top - bottom, left, top
        source_crop = (cw, ch, cx, cy) if cw > 0 and ch > 0 else None
        print(f"[editor] Manual crop: {vw}×{vh} → {cw}×{ch} at ({cx},{cy})", flush=True)
    else:
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
        # Map lyric pronunciation windows onto the VIDEO output timeline.
        #
        # The plan stores lyric timings as AUDIO positions (song_start/end).
        # ffmpeg's drawtext enable='between(t,...)' uses VIDEO output time.
        # These are only equal when the plan perfectly covers the full audio timeline
        # with no gaps.  We compute `offset = running_time - seg.song_start` and shift
        # all lyric timings so subtitles always appear while the correct scene is on screen,
        # even when earlier slots were missing and introduced a drift.
        subtitles: list[tuple[str, float, float]] = []
        if seg.song_start >= 0:
            offset = running_time - seg.song_start   # 0 when plan is complete; compensates gaps
            if abs(offset) > 0.05:
                print(
                    f"[editor] Clip {seg.scene_index}: audio_pos={seg.song_start:.3f}s "
                    f"video_pos={running_time:.3f}s  drift={offset:+.3f}s — subtitle timing corrected"
                )
            clip_song_start = seg.song_start
            clip_song_end   = seg.song_end

            # lyrics.json is the single source of truth for subtitle timing.
            # When the caller passes fresh lyrics we scan the full list rather than
            # relying on the plan's embedded lyric_lines, which may be stale if the
            # user edited lyrics.json after the plan was generated.
            if lyrics is not None:
                lyric_source = [
                    ll for ll in lyrics
                    if ll.end_time > ll.start_time            # skip non-displayable markers
                    and ll.start_time < clip_song_end         # overlaps this segment's window
                    and ll.end_time   > clip_song_start
                ]
            else:
                lyric_source = seg.lyric_lines

            for lyric in lyric_source:
                t0 = (max(clip_song_start, lyric.start_time) + offset)
                t1 = (min(clip_song_end,   lyric.end_time)   + offset)
                t0 = max(running_time, t0)
                t1 = min(running_time + clip_dur, t1)
                if t1 <= t0:
                    continue
                subtitles.append((lyric.text, t0, t1))
        else:
            # Legacy: infer song-start from the first lyric, then compute running_time offsets
            seg_t0 = seg.lyric_lines[0].start_time if seg.lyric_lines else 0.0
            for lyric in seg.lyric_lines:
                t_start = max(0.0, lyric.start_time - seg_t0)
                t_end   = min(clip_dur, lyric.end_time - seg_t0)
                if t_end <= t_start:
                    continue
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

    # Build credit insert lines from metadata (empty string = line not shown)
    if artist or song_title:
        _parts = [p for p in [artist, song_title] if p]
        insert_line1 = "Sound : " + " – ".join(_parts)
    else:
        insert_line1 = ""
    if film_name:
        insert_line2 = "Script : " + film_name + (f" – {director}" if director else "")
    else:
        insert_line2 = ""

    filter_complex = _build_filter_complex(
        clips, source_crop=source_crop,
        insert_line1=insert_line1, insert_line2=insert_line2,
        subtitle_font=subtitle_font or SUBTITLE_FONT,
        subtitle_fontsize=subtitle_fontsize or SUBTITLE_FONTSIZE,
        subtitle_italic=subtitle_italic,
        output_duration=audio_end,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[editor] Rendering {len(clips)} clips ({running_time:.1f}s total) → {output_path}...")

    # Audio fade: clamp durations so they never overlap (e.g. very short reels)
    aud_fade_in  = min(AUDIO_FADE_DURATION, audio_end / 4)
    aud_fade_out = min(AUDIO_FADE_DURATION, audio_end / 4)
    aud_fade_out_start = max(0.0, audio_end - aud_fade_out)

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "1:a",
        "-t", str(audio_end),
        # ── Audio fade in/out ──────────────────────────────────────────────────
        "-af", (
            f"afade=t=in:st=0:d={aud_fade_in:.3f},"
            f"afade=t=out:st={aud_fade_out_start:.3f}:d={aud_fade_out:.3f}"
        ),
        # ── Video: Instagram Reels recommended encoding ────────────────────────
        # H.264 High profile, CRF 18 for high quality
        # yuv420p is required — 4:2:0 is the only pixel format Instagram accepts
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-profile:v", "high", "-level:v", "4.2",
        "-pix_fmt", "yuv420p",
        "-maxrate", "8M", "-bufsize", "16M",
        "-r", str(OUTPUT_FPS),
        "-movflags", "+faststart",
        # ── Audio: AAC stereo 256 kbps at 44.1 kHz ────────────────────────────
        "-c:a", "aac", "-b:a", "256k", "-ar", "44100", "-ac", "2",
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
    insert_line1: str = "",
    insert_line2: str = "",
    subtitle_font: str = SUBTITLE_FONT,
    subtitle_fontsize: int = SUBTITLE_FONTSIZE,
    subtitle_italic: bool = False,
    output_duration: float = 0.0,
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
    #    Both bg and fg receive the same source crop so black bars never bleed through.
    #    bg: crop bars → scale to cover 9:16 → center-crop → Gaussian blur → slight darken
    #    fg: crop bars → scale to fit (lanczos) → unsharp → colour pop
    parts.append("[vcat]split=2[bgraw][fgraw]")

    _bg_scale = (
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},"
        f"gblur=sigma=20,"
        f"eq=brightness=-0.08"
    )
    _fg_chain = (
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
        f":force_original_aspect_ratio=decrease:flags=lanczos,"
        f"unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.4,"
        f"eq=saturation=1.1:contrast=1.04:gamma=1.01,"
        f"noise=alls=7:allf=t+u"
    )
    if source_crop:
        cw, ch, cx, cy = source_crop
        # Apply source crop to both layers so no black bars in either background or foreground
        parts.append(f"[bgraw]crop={cw}:{ch}:{cx}:{cy},{_bg_scale}[bg]")
        parts.append(f"[fgraw]crop={cw}:{ch}:{cx}:{cy},{_fg_chain}[fg]")
    else:
        parts.append(f"[bgraw]{_bg_scale}[bg]")
        parts.append(f"[fgraw]{_fg_chain}[fg]")
    parts.append("[bg][fg]overlay=x=(W-w)/2:y=(H-h)/2[vbg]")

    # 4. Drawtext: subtitles + source-credit insert, chained in one pass
    all_subs: list[tuple[str, float, float]] = [
        (text, t0, t1)
        for (_, _, subs) in clips
        for (text, t0, t1) in subs
    ]

    all_drawtext: list[str] = []

    effective_font = _resolve_italic_font(subtitle_font) if subtitle_italic else subtitle_font
    for text, t0, t1 in all_subs:
        safe = _escape_drawtext(text)
        all_drawtext.append(
            f"drawtext="
            f"fontfile='{effective_font}':"
            f"text='{safe}':"
            f"x=(w-tw)/2:"
            f"y=h*{SUBTITLE_Y_RATIO}-th/2:"
            f"fontsize={subtitle_fontsize}:"
            f"fontcolor={SUBTITLE_COLOR}:"
            f"bordercolor={SUBTITLE_BORDER_COLOR}:borderw={SUBTITLE_BORDER_WIDTH}:"
            f"shadowx=0:shadowy=2:shadowcolor=black@0.45:"
            f"enable='between(t\\,{t0:.3f}\\,{t1:.3f})'"
        )

    # Credit insert — top-left corner, fades in at INSERT_START_T, fades out before INSERT_END_T
    _fade_end_t      = INSERT_START_T + INSERT_FADE_T        # end of fade-in
    _fade_out_start  = INSERT_END_T   - INSERT_FADE_OUT_T    # start of fade-out
    _alpha_expr = (
        f"if(lt(t\\,{INSERT_START_T:.1f})\\,0\\,"
        f"if(lt(t\\,{_fade_end_t:.1f})\\,(t-{INSERT_START_T:.1f})/{INSERT_FADE_T:.1f}\\,"
        f"if(lt(t\\,{_fade_out_start:.1f})\\,1\\,"
        f"if(lt(t\\,{INSERT_END_T:.1f})\\,({INSERT_END_T:.1f}-t)/{INSERT_FADE_OUT_T:.1f}\\,0))))"
    )
    _line_gap = int(INSERT_FONTSIZE * 1.45)
    for i, line in enumerate([insert_line1, insert_line2]):
        if not line:
            continue
        safe = _escape_drawtext(line)
        all_drawtext.append(
            f"drawtext="
            f"fontfile='{INSERT_FONT}':"
            f"text='{safe}':"
            f"x=w*{INSERT_X_RATIO}:"
            f"y=h*{INSERT_Y_TOP_RATIO}+{i * _line_gap}:"
            f"fontsize={INSERT_FONTSIZE}:"
            f"fontcolor={INSERT_COLOR}:"
            f"bordercolor=black:borderw={INSERT_BORDER_WIDTH}:"
            f"shadowx=0:shadowy=2:shadowcolor=black@0.5:"
            f"fix_bounds=1:"
            f"alpha='{_alpha_expr}'"
        )

    # 5. Video fade-in / fade-out — clamped so they never overlap on short reels
    vid_fade = min(VIDEO_FADE_DURATION, output_duration / 4) if output_duration > 0 else VIDEO_FADE_DURATION
    vid_fade_out_start = max(0.0, output_duration - vid_fade) if output_duration > 0 else 0.0
    _vfade = (
        f"fade=t=in:st=0:d={vid_fade:.3f},"
        f"fade=t=out:st={vid_fade_out_start:.3f}:d={vid_fade:.3f}"
    )

    if all_drawtext:
        parts.append(f"[vbg]{','.join(all_drawtext)},{_vfade}[vout]")
    else:
        parts.append(f"[vbg]{_vfade}[vout]")

    return ";".join(parts)



def _resolve_italic_font(font_path: str) -> str:
    """Return the italic variant of a font file, or font_path if none can be found.

    Only file-path lookups are used.  Fontconfig patterns (e.g. 'Family:slant=100')
    look correct on paper but ffmpeg's filter-graph parser splits option values on ':'
    before drawtext can forward the string to fontconfig, causing a hard crash.
    For .ttc bundle fonts (Helvetica, Avenir Next, Futura…) the italic face lives
    inside the same file and can't be addressed by path — italic is silently skipped.
    """
    from pathlib import Path as _P
    p = _P(font_path)
    stem, suffix, parent = p.stem, p.suffix, p.parent

    candidate = parent / f"{stem} Italic{suffix}"
    if candidate.exists():
        print(f"[editor] Italic font: {candidate}", flush=True)
        return str(candidate)

    print(f"[editor] No italic variant for '{stem}' — rendering without italic", flush=True)
    return font_path


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

    # ── Run cropdetect at two thresholds; take the most-common rectangle ────
    # limit=16 catches pure black bars; limit=64 catches slightly non-black bars
    # (noise from compression, film grain, or near-black fade frames).
    # Running both passes and picking the modal result keeps false positives low.
    all_matches: list[tuple[str, str, str, str]] = []
    for limit in (16, 64):
        result = subprocess.run(
            [
                FFMPEG_BIN, "-i", str(video_path),
                "-vf", f"cropdetect=limit={limit}:round=2:skip=2",
                "-t", "90",
                "-f", "null", "/dev/null",
            ],
            capture_output=True, text=True,
        )
        all_matches.extend(re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", result.stderr))

    if not all_matches:
        print("[editor] cropdetect: no crop values found — skipping")
        return None

    # Use the modal (most-common) crop rectangle across both passes
    (cw_s, ch_s, cx_s, cy_s) = Counter(all_matches).most_common(1)[0][0]
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
