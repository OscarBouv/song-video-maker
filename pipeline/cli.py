"""song-video-maker — full pipeline and individual step CLI."""
import json
import re
from pathlib import Path
from typing import Annotated, Optional

import typer

from config import (
    TEMP_DIR,
    OUTPUTS_DIR,
    SCENE_THRESHOLD,
    MIN_SCENE_DURATION,
    N_FRAMES_PER_SCENE,
    OpenRouterModel,
    DEFAULT_ANALYZER_MODEL,
    DEFAULT_MATCHER_MODEL,
)
from models import Scene, LyricLine, MatchedSegment

app = typer.Typer(
    name="song-video-maker",
    help=(
        "Generate aesthetic 9:16 Instagram Reels from movie clips and songs.\n\n"
        "Run the full pipeline with 'run', or execute each step individually for testing."
    ),
    no_args_is_help=True,
)


# ── State paths ───────────────────────────────────────────────────────────────

STATE_DIR            = TEMP_DIR / "state"
SCENES_FILE          = STATE_DIR / "scenes.json"
LYRICS_FILE          = STATE_DIR / "lyrics.json"
LYRICS_READABLE_FILE = STATE_DIR / "lyrics_readable.txt"
AUDIO_PATH_FILE      = STATE_DIR / "audio_path.txt"
VIDEO_PATH_FILE      = STATE_DIR / "video_path.txt"


# ── State helpers ─────────────────────────────────────────────────────────────

def _save_scenes(scenes: list[Scene]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SCENES_FILE.write_text(json.dumps([s.model_dump() for s in scenes], indent=2))
    typer.echo(f"[state] Saved {len(scenes)} scenes → {SCENES_FILE}")


def _load_scenes() -> list[Scene]:
    _require(SCENES_FILE, "detect-scenes")
    return [Scene.model_validate(d) for d in json.loads(SCENES_FILE.read_text())]


def _save_lyrics(lyrics: list[LyricLine]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LYRICS_FILE.write_text(json.dumps([l.model_dump() for l in lyrics], indent=2))
    typer.echo(f"[state] Saved {len(lyrics)} lyric lines → {LYRICS_FILE}")

    # Also write a human-readable version so the chunks are easy to inspect
    lines = [
        f"LYRICS — {len(lyrics)} chunks",
        f"{'─' * 52}",
    ]
    for i, ll in enumerate(lyrics):
        dur = ll.end_time - ll.start_time
        # Mark zero-duration (never-display) lines clearly
        if dur <= 0:
            lines.append(f"  [{i:>3}]  {'—never display—':<12}  {ll.text}")
        else:
            lines.append(
                f"  [{i:>3}]  {ll.start_time:>6.2f}s – {ll.end_time:>6.2f}s  "
                f"({dur:.2f}s)  {ll.text}"
            )
    LYRICS_READABLE_FILE.write_text("\n".join(lines))
    typer.echo(f"[state] Readable lyrics   → {LYRICS_READABLE_FILE}")


def _load_lyrics() -> list[LyricLine]:
    _require(LYRICS_FILE, "extract-lyrics")
    return [LyricLine.model_validate(d) for d in json.loads(LYRICS_FILE.read_text())]


def _save_audio_path(path: Path) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_PATH_FILE.write_text(str(path))


def _load_audio_path() -> Path:
    _require(AUDIO_PATH_FILE, "download-audio")
    return Path(AUDIO_PATH_FILE.read_text().strip())


def _save_video_path(path: Path) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_PATH_FILE.write_text(str(path))


def _load_video_path() -> Path:
    _require(VIDEO_PATH_FILE, "download-video")
    return Path(VIDEO_PATH_FILE.read_text().strip())


def _require(path: Path, step: str) -> None:
    if not path.exists():
        typer.echo(f"Error: {path.name} not found. Run 'song-video-maker {step}' first.", err=True)
        raise typer.Exit(1)


def _slugify(film: str, song: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", f"{film}_{song}".lower()).strip("_")


def _next(cmd: str) -> None:
    typer.echo(f"\n→ Next:  song-video-maker {cmd}\n")


def _apply_lyrics_to_plan(
    segments: list[MatchedSegment],
) -> list[MatchedSegment]:
    """Override every lyric in the plan with the current lyrics.json.

    The plan owns *which scenes to use* (scene_index, trim offsets, song_start/song_end).
    Lyric text and timestamps always come from lyrics.json, so any refinement
    (Whisper re-run, manual tweaks) is picked up on the next render without
    regenerating the plan.

    Strategy:
    • New-format plans (song_start ≥ 0): time-based — each segment's lyrics are
      refreshed to all lyrics.json lines that overlap its song_start..song_end window.
    • Legacy plans (song_start = -1): positional — the i-th unique lyric slot maps
      to the i-th line in lyrics.json.
    """
    if not LYRICS_FILE.exists():
        return segments

    fresh_lyrics = _load_lyrics()

    # ── New format: time-based match ─────────────────────────────────────────
    if any(seg.song_start >= 0 for seg in segments):
        result = []
        for seg in segments:
            if seg.song_start < 0:
                result.append(seg)
                continue

            seg_lyrics = [
                ll for ll in fresh_lyrics
                if ll.end_time > ll.start_time        # displayable
                and ll.start_time < seg.song_end
                and ll.end_time   > seg.song_start
            ]
            result.append(MatchedSegment(
                scene_index=seg.scene_index,
                lyric_lines=seg_lyrics,
                scene_trim_start=seg.scene_trim_start,
                scene_trim_end=seg.scene_trim_end,
                song_start=seg.song_start,
                song_end=seg.song_end,
            ))
        typer.echo(f"[render] Applied lyrics.json ({len(fresh_lyrics)} lines) → plan (time-based match)")
        return result

    # ── Legacy format: positional match ──────────────────────────────────────
    seen: dict[str, int] = {}
    unique_keys: list[str] = []
    for seg in segments:
        for ll in seg.lyric_lines:
            key = ll.text.strip().lower()
            if key not in seen:
                seen[key] = len(unique_keys)
                unique_keys.append(key)

    if len(unique_keys) != len(fresh_lyrics):
        typer.echo(
            f"\n[render] ✗ Lyric count mismatch — plan has {len(unique_keys)} unique lyric slots "
            f"but lyrics.json has {len(fresh_lyrics)} lines.\n"
            f"  This usually means extract-lyrics was re-run (producing new chunks) without\n"
            f"  regenerating the plan.  The render will use the lyrics embedded in the plan\n"
            f"  (possibly old LRCLIB timestamps or unchunked text).\n"
            f"  → Fix: song-video-maker generate-plan --film '...' --song '...'\n"
            f"         then re-run render.\n",
            err=True,
        )
        return segments  # best-effort fallback: use whatever the plan already contains

    slot_to_fresh: dict[str, LyricLine] = {
        key: fresh_lyrics[i] for i, key in enumerate(unique_keys)
    }
    result = []
    for seg in segments:
        result.append(MatchedSegment(
            scene_index=seg.scene_index,
            lyric_lines=[slot_to_fresh[ll.text.strip().lower()] for ll in seg.lyric_lines],
            scene_trim_start=seg.scene_trim_start,
            scene_trim_end=seg.scene_trim_end,
        ))

    typer.echo(f"[render] Applied lyrics.json ({len(fresh_lyrics)} lines) → plan (positional match)")
    return result


# ── Full pipeline ─────────────────────────────────────────────────────────────

@app.command()
def run(
    video_url: Annotated[str, typer.Option(help="YouTube URL of the movie clip compilation")],
    song_url:  Annotated[str, typer.Option(help="YouTube URL of the song")],
    film:      Annotated[str, typer.Option(help="Film name  (e.g. 'Licorice Pizza')")],
    song:      Annotated[str, typer.Option(help="Song title (e.g. 'Call it Fate, Call it Karma')")],
    artist:         Annotated[Optional[str],   typer.Option(help="Artist name — improves LRCLIB lookup")] = None,
    video_start:    Annotated[Optional[float], typer.Option(help="Video window start (seconds)")] = None,
    video_end:      Annotated[Optional[float], typer.Option(help="Video window end   (seconds)")] = None,
    song_start:     Annotated[Optional[float], typer.Option(help="Song window start (seconds)")] = None,
    song_end:       Annotated[Optional[float], typer.Option(help="Song window end   (seconds)")] = None,
    n_frames:       Annotated[int,             typer.Option(help="Frames sampled per scene for vision analysis")] = N_FRAMES_PER_SCENE,
    characters:     Annotated[Optional[str],   typer.Option(help="Comma-separated main character names for richer scene descriptions (e.g. 'Alana Kane, Gary Valentine')")] = None,
    analyzer_model: Annotated[OpenRouterModel, typer.Option(help="Vision model for scene analysis")] = DEFAULT_ANALYZER_MODEL,
    matcher_model:  Annotated[OpenRouterModel, typer.Option(help="Text model for scene-to-lyric matching")] = DEFAULT_MATCHER_MODEL,
    refine_timing:  Annotated[bool, typer.Option(
        "--refine-timing/--no-refine-timing",
        help="Whisper word-level subtitle timing (default: on)",
    )] = True,
    render_only: Annotated[bool, typer.Option("--render-only", help="Skip straight to render using an existing plan")] = False,
) -> None:
    """Run the full pipeline end-to-end.

    Stops after plan generation so you can review and edit it, then re-run with --render-only.
    """
    from pipeline.downloader import download_video, download_audio
    from pipeline.scene_detector import detect_scenes
    from pipeline.frame_sampler import sample_all_scenes
    from pipeline.scene_analyzer import analyze_scenes
    from pipeline.lyrics_extractor import extract_lyrics
    from pipeline.matcher import generate_plan
    from pipeline.editor import render_video

    slug        = _slugify(film, song)
    plan_path   = OUTPUTS_DIR / f"{slug}_plan.json"
    output_path = OUTPUTS_DIR / f"{slug}_reel.mp4"

    # ── Render-only mode ──────────────────────────────────────────────────────
    if render_only:
        if not plan_path.exists():
            typer.echo(f"Error: plan not found at {plan_path}. Run without --render-only first.", err=True)
            raise typer.Exit(1)
        video_path = _load_video_path()
        audio_path = _load_audio_path()
        scenes   = detect_scenes(video_path)
        segments = [MatchedSegment.from_dict(d) for d in json.loads(plan_path.read_text())]
        segments = _apply_lyrics_to_plan(segments)
        render_video(segments, scenes, video_path, audio_path, output_path)
        return

    # ── Full pipeline ─────────────────────────────────────────────────────────
    typer.echo("\n── Step 1/6: Downloading video ──────────────────────────────")
    video_path = download_video(video_url, TEMP_DIR, start_sec=video_start, end_sec=video_end)

    typer.echo("\n── Step 2/6: Downloading audio ──────────────────────────────")
    audio_path = download_audio(song_url, TEMP_DIR, start_sec=song_start, end_sec=song_end, track=song, artist=artist)

    typer.echo("\n── Step 3/6: Detecting scenes ───────────────────────────────")
    scenes = detect_scenes(video_path)

    typer.echo("\n── Step 4/6: Sampling frames + AI scene analysis ────────────")
    scenes = sample_all_scenes(video_path, scenes, n_frames=n_frames)
    char_list = [c.strip() for c in characters.split(",")] if characters else None
    scenes = analyze_scenes(scenes, film_name=film, model=analyzer_model.value, characters=char_list)

    typer.echo("\n── Step 5/6: Extracting lyrics ──────────────────────────────")
    lyrics = extract_lyrics(
        audio_path, track=song, artist=artist,
        start_sec=song_start, end_sec=song_end,
        refine_timing=refine_timing,
    )
    if not lyrics:
        typer.echo("Error: no lyrics found. Check your audio or song title.", err=True)
        raise typer.Exit(1)

    typer.echo("\n── Step 6/6: Generating scene-to-lyric plan ─────────────────")
    generate_plan(
        scenes=scenes,
        lyrics=lyrics,
        film_name=film,
        song_name=song,
        output_dir=OUTPUTS_DIR,
        slug=slug,
        model=matcher_model.value,
        characters=char_list,
        audio_path=audio_path,
    )

    typer.echo("\n✓ Pipeline complete. Next steps:")
    typer.echo(f"  1. Review  : outputs/{slug}_plan_readable.txt")
    typer.echo(f"  2. Edit    : outputs/{slug}_plan.json  (if needed)")
    typer.echo(
        f"  3. Render  : song-video-maker run --film '{film}' --song '{song}'"
        f" --video-url '...' --song-url '...' --render-only"
    )


# ── Individual steps ──────────────────────────────────────────────────────────

@app.command("download-video")
def download_video_cmd(
    url:         Annotated[str,             typer.Option(help="YouTube URL of the clip compilation")],
    video_start: Annotated[Optional[float], typer.Option(help="Start trim (seconds)")] = None,
    video_end:   Annotated[Optional[float], typer.Option(help="End trim   (seconds)")] = None,
) -> None:
    """Download the source video compilation, optionally pre-cut to a time window."""
    from pipeline.downloader import download_video
    path = download_video(url, TEMP_DIR, start_sec=video_start, end_sec=video_end)
    _save_video_path(path)
    typer.echo(f"\n✓ Video: {path}")
    _next("download-audio --url '<song_url>'")


@app.command("download-audio")
def download_audio_cmd(
    url:        Annotated[str,             typer.Option(help="YouTube URL of the song")],
    song:       Annotated[Optional[str],   typer.Option(help="Song title (used for filename)")] = None,
    artist:     Annotated[Optional[str],   typer.Option(help="Artist name (used for filename)")] = None,
    song_start: Annotated[Optional[float], typer.Option(help="Start trim (seconds)")] = None,
    song_end:   Annotated[Optional[float], typer.Option(help="End trim   (seconds)")] = None,
) -> None:
    """Download song audio, optionally trimmed to a time window."""
    from pipeline.downloader import download_audio
    path = download_audio(url, TEMP_DIR, start_sec=song_start, end_sec=song_end, track=song, artist=artist)
    _save_audio_path(path)
    typer.echo(f"\n✓ Audio: {path}")
    _next("detect-scenes  (and separately)  song-video-maker extract-lyrics --song '...'")


@app.command("detect-scenes")
def detect_scenes_cmd(
    threshold:    Annotated[float, typer.Option(help="PySceneDetect sensitivity")] = SCENE_THRESHOLD,
    min_duration: Annotated[float, typer.Option(help="Minimum scene length (seconds)")] = MIN_SCENE_DURATION,
) -> None:
    """Detect scene cuts in the downloaded video → saves temp/state/scenes.json."""
    from pipeline.scene_detector import detect_scenes
    video_path = _load_video_path()
    scenes = detect_scenes(video_path, threshold=threshold, min_duration=min_duration)
    _print_scenes_table(scenes)
    _save_scenes(scenes)
    typer.echo(f"\n✓ {len(scenes)} scenes detected")
    _next("sample-frames")


@app.command("sample-frames")
def sample_frames_cmd(
    n_frames: Annotated[int, typer.Option(help="Frames to extract per scene")] = N_FRAMES_PER_SCENE,
) -> None:
    """Extract representative frames from each scene → updates scenes.json."""
    from pipeline.frame_sampler import sample_all_scenes
    video_path = _load_video_path()
    scenes = _load_scenes()
    scenes = sample_all_scenes(video_path, scenes, n_frames=n_frames)
    _save_scenes(scenes)
    typer.echo(f"\n✓ Sampled {n_frames} frames × {len(scenes)} scenes")
    _next("analyze-scenes --film '<film>'")


@app.command("analyze-scenes")
def analyze_scenes_cmd(
    film:        Annotated[str,             typer.Option(help="Film name used for filtering unrelated scenes")],
    characters:  Annotated[Optional[str],   typer.Option(help="Comma-separated main character names (e.g. 'Alana Kane, Gary Valentine')")] = None,
    model:       Annotated[OpenRouterModel, typer.Option(help="Vision model to use")] = DEFAULT_ANALYZER_MODEL,
    max_batches: Annotated[Optional[int],   typer.Option(help="Stop after N batches — useful for testing descriptions on a small sample")] = None,
) -> None:
    """Describe each scene with a vision LLM and flag non-film scenes → updates scenes.json."""
    from pipeline.scene_analyzer import analyze_scenes
    scenes = _load_scenes()
    missing = [s for s in scenes if not s.frames]
    if missing:
        typer.echo(f"Error: {len(missing)} scenes have no frames. Run 'sample-frames' first.", err=True)
        raise typer.Exit(1)
    char_list = [c.strip() for c in characters.split(",")] if characters else None
    scenes = analyze_scenes(scenes, film_name=film, model=model.value, characters=char_list, max_batches=max_batches)
    _save_scenes(scenes)
    related   = sum(1 for s in scenes if s.is_film_related)
    aesthetic = sum(1 for s in scenes if s.is_aesthetic)
    usable    = sum(1 for s in scenes if s.is_usable)
    typer.echo(f"\n✓ {related}/{len(scenes)} film-related  |  {aesthetic}/{len(scenes)} aesthetic  |  {usable} usable for matching")
    _next("generate-plan --film '<film>' --song '<song>'")


@app.command("extract-lyrics")
def extract_lyrics_cmd(
    song:          Annotated[str,             typer.Option(help="Song title")],
    artist:        Annotated[Optional[str],   typer.Option(help="Artist name (improves LRCLIB lookup)")] = None,
    song_start:    Annotated[Optional[float], typer.Option(help="Window start (seconds)")] = None,
    song_end:      Annotated[Optional[float], typer.Option(help="Window end   (seconds)")] = None,
    refine_timing: Annotated[bool,            typer.Option(
        "--refine-timing/--no-refine-timing",
        help="Use Whisper word-level timestamps to show subtitles only while singing (default: on)",
    )] = True,
) -> None:
    """Fetch synced lyrics via LRCLIB (+ Whisper timing refinement) → saves temp/state/lyrics.json."""
    from pipeline.lyrics_extractor import extract_lyrics
    audio_path = _load_audio_path()
    lyrics = extract_lyrics(
        audio_path, track=song, artist=artist,
        start_sec=song_start, end_sec=song_end,
        refine_timing=refine_timing,
    )
    if not lyrics:
        typer.echo("Error: no lyrics found.", err=True)
        raise typer.Exit(1)
    _save_lyrics(lyrics)

    # ── Print a readable preview in the terminal ──────────────────────────────
    preview_n = min(len(lyrics), 12)
    typer.echo(f"\n── Lyric chunks ({len(lyrics)} total) ────────────────────────────────")
    non_zero = [ll for ll in lyrics if ll.end_time > ll.start_time]
    for ll in lyrics[:preview_n]:
        dur = ll.end_time - ll.start_time
        if dur <= 0:
            typer.echo(f"  {'—':>14}  {ll.text}")
        else:
            typer.echo(f"  {ll.start_time:>6.2f}–{ll.end_time:>6.2f}s  {ll.text}")
    if len(lyrics) > preview_n:
        typer.echo(f"  … ({len(lyrics) - preview_n} more — see {LYRICS_READABLE_FILE.name})")

    total_dur = non_zero[-1].end_time if non_zero else 0.0
    typer.echo(
        f"\n✓ {len(lyrics)} lyric chunks  |  "
        f"{len(non_zero)} displayable  |  "
        f"song window: 0 – {total_dur:.1f}s"
    )
    typer.echo(f"  Full list: {LYRICS_READABLE_FILE}")

    # ── Warn if any existing plan.json won't match the new lyric count ────────
    plan_files = list(OUTPUTS_DIR.glob("*_plan.json"))
    stale = []
    for pf in plan_files:
        try:
            plan_data = json.loads(pf.read_text())
            plan_lyric_texts = {
                ll["text"].strip().lower()
                for seg in plan_data
                for ll in seg.get("lyric_lines", [])
            }
            if len(plan_lyric_texts) != len(lyrics):
                stale.append(pf.name)
        except Exception:
            pass
    if stale:
        typer.echo(
            f"\n⚠️  {len(stale)} plan file(s) have a different lyric count and will NOT "
            f"sync correctly on render:\n"
            + "\n".join(f"     {n}" for n in stale)
            + f"\n   → Re-run: song-video-maker generate-plan --film '...' --song '...'"
        )

    _next("generate-plan --film '<film>' --song '<song>'")


@app.command("generate-plan")
def generate_plan_cmd(
    film:       Annotated[str,             typer.Option(help="Film name")],
    song:       Annotated[str,             typer.Option(help="Song title")],
    model:      Annotated[OpenRouterModel, typer.Option(help="Text model to use")] = DEFAULT_MATCHER_MODEL,
    characters: Annotated[Optional[str],   typer.Option(help="Comma-separated main character names (e.g. 'Alana Kane, Gary Valentine')")] = None,
) -> None:
    """Ask an LLM to map scenes to lyrics → saves outputs/{slug}_plan.json + _readable.txt."""
    from pipeline.matcher import generate_plan
    scenes = _load_scenes()
    lyrics = _load_lyrics()
    audio_path = _load_audio_path()   # needed for exact audio duration
    slug = _slugify(film, song)
    char_list = [c.strip() for c in characters.split(",")] if characters else None
    generate_plan(
        scenes=scenes,
        lyrics=lyrics,
        film_name=film,
        song_name=song,
        output_dir=OUTPUTS_DIR,
        slug=slug,
        model=model.value,
        characters=char_list,
        audio_path=audio_path,
    )
    typer.echo(f"\n✓ Plan: outputs/{slug}_plan_readable.txt")
    _next(f"render --film '{film}' --song '{song}'")


@app.command()
def render(
    film: Annotated[str, typer.Option(help="Film name")],
    song: Annotated[str, typer.Option(help="Song title")],
) -> None:
    """Assemble the final 9:16 Reel from the plan → outputs/{slug}_reel.mp4."""
    from pipeline.editor import render_video
    video_path = _load_video_path()
    audio_path = _load_audio_path()
    scenes = _load_scenes()
    slug = _slugify(film, song)
    plan_path = OUTPUTS_DIR / f"{slug}_plan.json"
    _require(plan_path, f"generate-plan --film '{film}' --song '{song}'")
    segments = [MatchedSegment.from_dict(d) for d in json.loads(plan_path.read_text())]
    segments = _apply_lyrics_to_plan(segments)
    output_path = OUTPUTS_DIR / f"{slug}_reel.mp4"
    render_video(segments, scenes, video_path, audio_path, output_path)
    typer.echo(f"\n✓ Reel: {output_path}")


@app.command()
def status() -> None:
    """Show current state of all pipeline files."""
    def _row(label: str, path: Path, detail: str = "") -> None:
        mark = typer.style("✓", fg=typer.colors.GREEN) if path.exists() else typer.style("✗", fg=typer.colors.RED)
        size = f"({path.stat().st_size // 1024}KB)" if path.exists() else ""
        typer.echo(f"  {mark}  {label:<26} {size:<10} {detail}")

    typer.echo("\n── Downloads ─────────────────────────────────────────────────")
    video_display = Path(VIDEO_PATH_FILE.read_text().strip()) if VIDEO_PATH_FILE.exists() else TEMP_DIR / "video.mp4"
    _row(f"temp/{video_display.name}", video_display)
    audio_display = Path(AUDIO_PATH_FILE.read_text().strip()) if AUDIO_PATH_FILE.exists() else TEMP_DIR / "audio.mp3"
    _row(f"temp/{audio_display.name}", audio_display)

    typer.echo("\n── State ─────────────────────────────────────────────────────")
    if SCENES_FILE.exists():
        scenes   = [Scene.model_validate(d) for d in json.loads(SCENES_FILE.read_text())]
        related  = sum(1 for s in scenes if s.is_film_related)
        n_frames = sum(1 for s in scenes if s.frames)
        n_desc   = sum(1 for s in scenes if s.description)
        aesthetic = sum(1 for s in scenes if s.is_aesthetic)
        usable    = sum(1 for s in scenes if s.is_usable)
        detail = f"{len(scenes)} scenes  {n_frames} w/frames  {n_desc} analyzed  {related} film-related  {aesthetic} aesthetic  {usable} usable"
        _row("temp/state/scenes.json", SCENES_FILE, detail)
    else:
        _row("temp/state/scenes.json", SCENES_FILE)

    if LYRICS_FILE.exists():
        lyrics  = [LyricLine.model_validate(d) for d in json.loads(LYRICS_FILE.read_text())]
        preview = (lyrics[0].text[:40] + "...") if lyrics else ""
        detail  = f'{len(lyrics)} lines  "{preview}"'
        _row("temp/state/lyrics.json", LYRICS_FILE, detail)
    else:
        _row("temp/state/lyrics.json", LYRICS_FILE)

    typer.echo("\n── Outputs ───────────────────────────────────────────────────")
    for f in sorted(OUTPUTS_DIR.glob("*")):
        _row(f"outputs/{f.name}", f)

    typer.echo()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_scenes_table(scenes: list[Scene]) -> None:
    typer.echo(f"\n{'#':>4}  {'Start':>8}  {'End':>8}  {'Dur':>6}")
    typer.echo("  " + "─" * 34)
    for s in scenes[:20]:
        typer.echo(f"  {s.index:>4}  {s.start_time:>7.1f}s  {s.end_time:>7.1f}s  {s.duration:>5.1f}s")
    if len(scenes) > 20:
        typer.echo(f"  ... ({len(scenes) - 20} more)")


if __name__ == "__main__":
    app()
