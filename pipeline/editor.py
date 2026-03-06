"""Assemble the final 9:16 Instagram Reel from matched segments + audio."""
from pathlib import Path

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

from config import (
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    OUTPUT_FPS,
    MAX_REEL_DURATION,
    SUBTITLE_COLOR,
    SUBTITLE_FONTSIZE,
    SUBTITLE_FONT,
    SUBTITLE_POSITION,
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
    """Assemble clips, add subtitles, mix audio, export 9:16 MP4."""
    scene_by_index = {s.index: s for s in scenes}
    source_video = VideoFileClip(str(video_path))

    clip_segments: list[CompositeVideoClip] = []
    running_time = 0.0

    for seg in segments:
        scene = scene_by_index.get(seg.scene_index)
        if scene is None:
            print(f"[editor] Warning: scene {seg.scene_index} not found, skipping")
            continue

        # Clip bounds within source video
        clip_start = scene.start_time + seg.scene_trim_start
        if seg.scene_trim_end >= 0:
            clip_end = scene.start_time + seg.scene_trim_end
        else:
            clip_end = scene.end_time

        clip_end = min(clip_end, source_video.duration)
        clip_dur = clip_end - clip_start

        if clip_dur <= 0:
            print(f"[editor] Warning: scene {seg.scene_index} has zero duration, skipping")
            continue

        if running_time + clip_dur > max_duration:
            clip_dur = max_duration - running_time
            clip_end = clip_start + clip_dur
            if clip_dur <= 0:
                break

        # Extract and crop to 9:16
        clip = source_video.subclipped(clip_start, clip_end)
        clip = _crop_to_vertical(clip)

        # Build subtitle overlays for this clip
        subtitle_clips: list[TextClip] = []
        for lyric in seg.lyric_lines:
            # Translate song-absolute timestamp to position within this clip
            lyric_clip_start = lyric.start_time - _segment_song_start(seg)
            lyric_clip_end = lyric.end_time - _segment_song_start(seg)

            # Clamp to clip bounds
            lyric_clip_start = max(0.0, lyric_clip_start)
            lyric_clip_end = min(clip_dur, lyric_clip_end)
            if lyric_clip_end <= lyric_clip_start:
                lyric_clip_start = 0.0
                lyric_clip_end = clip_dur

            txt = TextClip(
                text=lyric.text,
                font=SUBTITLE_FONT,
                font_size=SUBTITLE_FONTSIZE,
                color=SUBTITLE_COLOR,
                stroke_color="black",
                stroke_width=2,
                method="label",
                size=(OUTPUT_WIDTH - 80, None),
            )
            txt = txt.with_start(lyric_clip_start).with_end(lyric_clip_end)
            # Position: center horizontally, 80% down vertically
            x_pos = (OUTPUT_WIDTH - txt.w) // 2
            y_pos = int(OUTPUT_HEIGHT * SUBTITLE_POSITION[1])
            txt = txt.with_position((x_pos, y_pos))
            subtitle_clips.append(txt)

        composite = CompositeVideoClip([clip] + subtitle_clips)
        clip_segments.append(composite)
        running_time += clip_dur

        if running_time >= max_duration:
            break

    source_video.close()

    if not clip_segments:
        raise RuntimeError("[editor] No clips to render — check your plan and scenes")

    print(f"[editor] Concatenating {len(clip_segments)} clips ({running_time:.1f}s total)...")
    final = concatenate_videoclips(clip_segments, method="compose")

    # Mix in song audio
    audio = AudioFileClip(str(audio_path)).subclipped(0, running_time)
    final = final.with_audio(audio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[editor] Rendering to {output_path}...")
    final.write_videofile(
        str(output_path),
        fps=OUTPUT_FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(output_path.parent / "temp_audio.m4a"),
        remove_temp=True,
        logger="bar",
    )
    print(f"[editor] Done! Output: {output_path}")


def _crop_to_vertical(clip: VideoFileClip) -> VideoFileClip:
    """Center-crop clip to 9:16 (1080x1920). Scales up if needed."""
    target_w, target_h = OUTPUT_WIDTH, OUTPUT_HEIGHT
    src_w, src_h = clip.size

    # Scale so height matches target, then crop width (portrait source)
    scale = target_h / src_h
    new_w = int(src_w * scale)
    new_h = target_h

    if new_w < target_w:
        # Source is very wide — scale by width instead and add black bars or crop height
        scale = target_w / src_w
        new_w = target_w
        new_h = int(src_h * scale)

    clip = clip.resized((new_w, new_h))

    # Center crop
    x1 = (new_w - target_w) // 2
    y1 = (new_h - target_h) // 2
    clip = clip.cropped(x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)
    return clip


def _segment_song_start(seg: MatchedSegment) -> float:
    """Return the song timestamp at which this segment starts (from first lyric)."""
    if seg.lyric_lines:
        return seg.lyric_lines[0].start_time
    return 0.0
