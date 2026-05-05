"""song-video-maker — full pipeline and individual step CLI.

Every command operates on a **workspace** — an isolated directory named
after the film+song combination (e.g. workspaces/licorice_pizza_call_it_fate/).
All downloads, frames, cache, state, plan files and the final reel live there.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import typer

from config import (
    WORKSPACES_DIR,
    CLIPS_DIR,
    SONGS_DIR,
    SCENE_THRESHOLD,
    MIN_SCENE_DURATION,
    N_FRAMES_PER_SCENE,
    OpenRouterModel,
    DEFAULT_ANALYZER_MODEL,
    DEFAULT_MATCHER_MODEL,
)
from pipeline.plan_editor import DEFAULT_EDIT_MODEL
from models import Scene, LyricLine, MatchedSegment

app = typer.Typer(
    name="song-video-maker",
    help=(
        "Generate aesthetic 9:16 Instagram Reels from movie clips and songs.\n\n"
        "Each film+song combination gets its own workspace under workspaces/.\n"
        "Run the full pipeline with 'run', or execute each step individually."
    ),
    no_args_is_help=True,
)


# ── Slug helpers ─────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    """Slugify a single string."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _slugify(film: str, song: str) -> str:
    """Workspace slug (film+song, unchanged for backward compat)."""
    return _slug(f"{film}_{song}")


# ── Workspace paths ───────────────────────────────────────────────────────────

class _WS:
    """File paths for one workspace — a specific clip+song pairing.

    Data is stored in three disjoint directories:
      clips/{clip_slug}/     — video file, scenes, frames, scene-analysis cache
      songs/{song_slug}/     — audio file, lyrics, lyrics-readable
      workspaces/{slug}/     — plan, reel, LLM-response cache (the pairing)

    clip_slug and song_slug are derived from film/artist/song names and stored
    in workspace.json so any subsequent command can locate the right dirs even
    when the artist is not passed on the command line.
    """

    def __init__(self, film: str, song: str, artist: str = "") -> None:
        self.film   = film
        self.song   = song
        self.artist = artist
        self.slug   = _slugify(film, song)       # workspace dir name (unchanged)

        self.root        = WORKSPACES_DIR / self.slug
        self.config_file = self.root / "workspace.json"

        # Load stored slugs from workspace.json if available (set during migration
        # or first run) so we always resolve the right clip/song dir.
        cfg: dict = {}
        if self.config_file.exists():
            try:
                cfg = json.loads(self.config_file.read_text())
            except Exception:
                pass
        effective_artist = artist or cfg.get("artist", "")

        self.clip_slug = cfg.get("clip_slug") or _slug(film)
        self.song_slug = cfg.get("song_slug") or (
            _slug(f"{effective_artist}_{song}") if effective_artist else _slug(song)
        )

        # ── Clip-scoped dirs/files (clips/{clip_slug}/) ──────────────────
        self.clip_dir        = CLIPS_DIR / self.clip_slug
        self.frames_dir      = self.clip_dir / "frames"
        self.clip_cache_dir  = self.clip_dir / "cache"
        self.scenes_file     = self.clip_dir / "scenes.json"
        self.video_path_file = self.clip_dir / "video_path.txt"

        # ── Song-scoped dirs/files (songs/{song_slug}/) ──────────────────
        self.song_dir             = SONGS_DIR / self.song_slug
        self.lyrics_file          = self.song_dir / "lyrics.json"
        self.lyrics_readable_file = self.song_dir / "lyrics_readable.txt"
        self.audio_path_file      = self.song_dir / "audio_path.txt"

        # ── Workspace-scoped files (workspaces/{slug}/) ──────────────────
        self.plan_cache_dir     = self.root / "cache"   # LLM response cache
        self.plan_file          = self.root / f"{self.slug}_plan.json"
        self.plan_readable_file = self.root / f"{self.slug}_plan_readable.txt"
        self.reel_file          = self.root / f"{self.slug}_reel.mp4"

    def ensure(self) -> None:
        """Create all required directories."""
        for d in (self.clip_dir, self.clip_cache_dir, self.frames_dir,
                  self.song_dir, self.root, self.plan_cache_dir):
            d.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _next(cmd: str) -> None:
    typer.echo(f"\n→ Next:  song-video-maker {cmd}\n")


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _update_config(ws: _WS, **fields) -> None:
    """Merge fields into workspace.json, preserving any existing values."""
    ws.root.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if ws.config_file.exists():
        try:
            existing = json.loads(ws.config_file.read_text())
        except Exception:
            pass
    # Strip None values so they don't overwrite previously set fields
    updates = {k: v for k, v in fields.items() if v is not None}
    existing.update(updates)
    # Always keep core identity + slug mappings so any command can locate the right dirs
    existing.setdefault("film",      ws.film)
    existing.setdefault("song",      ws.song)
    existing.setdefault("clip_slug", ws.clip_slug)
    existing.setdefault("song_slug", ws.song_slug)
    existing.setdefault("created_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    ws.config_file.write_text(json.dumps(existing, indent=2))


def _require(path: Path, step: str, ws: _WS) -> None:
    if not path.exists():
        typer.echo(
            f"Error: {path.name} not found. Run 'song-video-maker {step}"
            f" --film \"{ws.film}\" --song \"{ws.song}\"' first.",
            err=True,
        )
        raise typer.Exit(1)


# ── State helpers (all workspace-scoped) ──────────────────────────────────────

def _save_scenes(ws: _WS, scenes: list[Scene]) -> None:
    ws.ensure()
    ws.scenes_file.write_text(json.dumps([s.model_dump() for s in scenes], indent=2))
    typer.echo(f"[state] Saved {len(scenes)} scenes → {ws.scenes_file}")


def _load_scenes(ws: _WS) -> list[Scene]:
    _require(ws.scenes_file, "detect-scenes", ws)
    return [Scene.model_validate(d) for d in json.loads(ws.scenes_file.read_text())]


def _save_lyrics(ws: _WS, lyrics: list[LyricLine]) -> None:
    ws.ensure()
    ws.lyrics_file.write_text(json.dumps([ll.model_dump() for ll in lyrics], indent=2))
    typer.echo(f"[state] Saved {len(lyrics)} lyric lines → {ws.lyrics_file}")

    lines = [
        f"LYRICS — {len(lyrics)} chunks",
        f"{'─' * 52}",
    ]
    for i, ll in enumerate(lyrics):
        dur = ll.end_time - ll.start_time
        if dur <= 0:
            lines.append(f"  [{i:>3}]  {'—never display—':<12}  {ll.text}")
        else:
            lines.append(
                f"  [{i:>3}]  {ll.start_time:>6.2f}s – {ll.end_time:>6.2f}s  "
                f"({dur:.2f}s)  {ll.text}"
            )
    ws.lyrics_readable_file.write_text("\n".join(lines))
    typer.echo(f"[state] Readable lyrics   → {ws.lyrics_readable_file}")


def _load_lyrics(ws: _WS) -> list[LyricLine]:
    _require(ws.lyrics_file, "extract-lyrics", ws)
    return [LyricLine.model_validate(d) for d in json.loads(ws.lyrics_file.read_text())]


def _load_lyrics_safe(ws: _WS) -> list[LyricLine]:
    """Load lyrics.json without raising — returns [] if missing or corrupt."""
    if not ws.lyrics_file.exists():
        return []
    try:
        return [LyricLine.model_validate(d) for d in json.loads(ws.lyrics_file.read_text())]
    except Exception as e:
        typer.echo(f"[render] Warning: could not load lyrics.json ({e})", err=True)
        return []


def _save_audio_path(ws: _WS, path: Path) -> None:
    ws.ensure()
    ws.audio_path_file.write_text(str(path))


def _load_audio_path(ws: _WS) -> Path:
    _require(ws.audio_path_file, "download-audio", ws)
    return Path(ws.audio_path_file.read_text().strip())


def _save_video_path(ws: _WS, path: Path) -> None:
    ws.ensure()
    ws.video_path_file.write_text(str(path))


def _load_video_path(ws: _WS) -> Path:
    _require(ws.video_path_file, "download-video", ws)
    return Path(ws.video_path_file.read_text().strip())


def _apply_lyrics_to_plan(
    segments: list[MatchedSegment],
    ws: _WS,
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
    if not ws.lyrics_file.exists():
        return segments

    fresh_lyrics = _load_lyrics(ws)

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
            f"  → Fix: song-video-maker generate-plan --film '{ws.film}' --song '{ws.song}'\n"
            f"         then re-run render.\n",
            err=True,
        )
        return segments  # best-effort fallback

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
    characters:     Annotated[Optional[str],   typer.Option(help="Comma-separated main character names (e.g. 'Alana Kane, Gary Valentine')")] = None,
    analyzer_model: Annotated[OpenRouterModel, typer.Option(help="Vision model for scene analysis")] = DEFAULT_ANALYZER_MODEL,
    matcher_model:  Annotated[OpenRouterModel, typer.Option(help="Text model for scene-to-lyric matching")] = DEFAULT_MATCHER_MODEL,
    refine_timing:  Annotated[bool, typer.Option(
        "--refine-timing/--no-refine-timing",
        help="Whisper word-level subtitle timing (default: on)",
    )] = True,
    render_only: Annotated[bool, typer.Option("--render-only", help="Skip straight to render using an existing plan")] = False,
    publish: Annotated[bool, typer.Option("--publish", help="Publish to Instagram after rendering")] = False,
) -> None:
    """Run the full pipeline end-to-end.

    All outputs go to workspaces/{slug}/. Stops after plan generation so you can
    review and edit it, then re-run with --render-only.
    """
    from pipeline.downloader import download_video, download_audio
    from pipeline.scene_detector import detect_scenes
    from pipeline.frame_sampler import sample_all_scenes
    from pipeline.scene_analyzer import analyze_scenes
    from pipeline.lyrics_extractor import extract_lyrics
    from pipeline.matcher import generate_plan
    from pipeline.editor import render_video

    ws = _WS(film, song)
    _update_config(
        ws,
        video_url=video_url, song_url=song_url, artist=artist,
        video_start=video_start, video_end=video_end,
        song_start=song_start, song_end=song_end,
    )

    # ── Render-only mode ──────────────────────────────────────────────────────
    if render_only:
        _require(ws.plan_file, f"generate-plan", ws)
        cfg = json.loads(ws.config_file.read_text()) if ws.config_file.exists() else {}
        video_path   = _load_video_path(ws)
        audio_path   = _load_audio_path(ws)
        scenes       = _load_scenes(ws)
        fresh_lyrics = _load_lyrics_safe(ws)
        segments     = [MatchedSegment.from_dict(d) for d in json.loads(ws.plan_file.read_text())]
        render_video(
            segments, scenes, video_path, audio_path, ws.reel_file,
            artist=cfg.get("artist", artist or ""),
            song_title=song,
            film_name=film,
            director=cfg.get("director", ""),
            lyrics=fresh_lyrics or None,
        )
        typer.echo(f"\n✓ Reel: {ws.reel_file}")

        if publish:
            from pipeline.publisher import publish_to_instagram
            default_caption = f"'{song}' by {film} (AI generated reel) #aesthetic #reels"
            publish_to_instagram(ws.reel_file, default_caption)
        return

    # ── Full pipeline ─────────────────────────────────────────────────────────
    ws.ensure()
    typer.echo(f"\n── Workspace: {ws.root}\n")

    typer.echo("\n── Step 1/6: Downloading video ──────────────────────────────")
    video_path = download_video(video_url, ws.root, start_sec=video_start, end_sec=video_end)
    _save_video_path(ws, video_path)

    typer.echo("\n── Step 2/6: Downloading audio ──────────────────────────────")
    audio_path = download_audio(song_url, ws.root, start_sec=song_start, end_sec=song_end, track=song, artist=artist)
    _save_audio_path(ws, audio_path)

    typer.echo("\n── Step 3/6: Detecting scenes ───────────────────────────────")
    scenes = detect_scenes(video_path)

    typer.echo("\n── Step 4/6: Sampling frames + AI scene analysis ────────────")
    scenes = sample_all_scenes(video_path, scenes, n_frames=n_frames, frames_dir=ws.frames_dir)
    char_list = [c.strip() for c in characters.split(",")] if characters else None
    scenes = analyze_scenes(scenes, film_name=film, model=analyzer_model.value, characters=char_list, cache_dir=ws.clip_cache_dir)
    _save_scenes(ws, scenes)

    typer.echo("\n── Step 5/6: Extracting lyrics ──────────────────────────────")
    lyrics = extract_lyrics(
        audio_path, track=song, artist=artist,
        start_sec=song_start, end_sec=song_end,
        refine_timing=refine_timing,
    )
    if not lyrics:
        typer.echo("Error: no lyrics found. Check your audio or song title.", err=True)
        raise typer.Exit(1)
    _save_lyrics(ws, lyrics)

    typer.echo("\n── Step 6/6: Generating scene-to-lyric plan ─────────────────")
    generate_plan(
        scenes=scenes,
        lyrics=lyrics,
        film_name=film,
        song_name=song,
        output_dir=ws.root,
        slug=ws.slug,
        model=matcher_model.value,
        characters=char_list,
        audio_path=audio_path,
        cache_dir=ws.plan_cache_dir,
    )

    typer.echo("\n✓ Pipeline complete. Next steps:")
    typer.echo(f"  1. Review  : {ws.plan_readable_file}")
    typer.echo(f"  2. Edit    : {ws.plan_file}  (if needed)")
    typer.echo(
        f"  3. Render  : song-video-maker run --film '{film}' --song '{song}'"
        f" --video-url '...' --song-url '...' --render-only"
    )


# ── Individual steps ──────────────────────────────────────────────────────────

@app.command("download-video")
def download_video_cmd(
    url:         Annotated[str,             typer.Option(help="YouTube URL of the clip compilation")],
    film:        Annotated[str,             typer.Option(help="Film name")],
    song:        Annotated[str,             typer.Option(help="Song title")],
    video_start: Annotated[Optional[float], typer.Option(help="Start trim (seconds)")] = None,
    video_end:   Annotated[Optional[float], typer.Option(help="End trim   (seconds)")] = None,
) -> None:
    """Download the source video compilation into the workspace."""
    from pipeline.downloader import download_video
    ws = _WS(film, song)
    ws.ensure()
    _update_config(ws, video_url=url, video_start=video_start, video_end=video_end)
    path = download_video(url, ws.root, start_sec=video_start, end_sec=video_end)
    _save_video_path(ws, path)
    typer.echo(f"\n✓ Video: {path}")
    _next(f"download-audio --url '<song_url>' --film '{film}' --song '{song}'")


@app.command("download-audio")
def download_audio_cmd(
    url:        Annotated[str,             typer.Option(help="YouTube URL of the song")],
    film:       Annotated[str,             typer.Option(help="Film name")],
    song:       Annotated[str,             typer.Option(help="Song title")],
    artist:     Annotated[Optional[str],   typer.Option(help="Artist name (used for filename)")] = None,
    song_start: Annotated[Optional[float], typer.Option(help="Start trim (seconds)")] = None,
    song_end:   Annotated[Optional[float], typer.Option(help="End trim   (seconds)")] = None,
) -> None:
    """Download song audio into the workspace."""
    from pipeline.downloader import download_audio
    ws = _WS(film, song)
    ws.ensure()
    _update_config(ws, song_url=url, artist=artist, song_start=song_start, song_end=song_end)
    path = download_audio(url, ws.root, start_sec=song_start, end_sec=song_end, track=song, artist=artist)
    _save_audio_path(ws, path)
    typer.echo(f"\n✓ Audio: {path}")
    _next(f"detect-scenes --film '{film}' --song '{song}'  (and)  song-video-maker extract-lyrics --film '{film}' --song '{song}'")


@app.command("detect-scenes")
def detect_scenes_cmd(
    film:         Annotated[str,   typer.Option(help="Film name")],
    song:         Annotated[str,   typer.Option(help="Song title")],
    threshold:    Annotated[float, typer.Option(help="PySceneDetect sensitivity")] = SCENE_THRESHOLD,
    min_duration: Annotated[float, typer.Option(help="Minimum scene length (seconds)")] = MIN_SCENE_DURATION,
) -> None:
    """Detect scene cuts in the downloaded video → saves state/scenes.json."""
    from pipeline.scene_detector import detect_scenes
    ws = _WS(film, song)
    video_path = _load_video_path(ws)
    scenes = detect_scenes(video_path, threshold=threshold, min_duration=min_duration)
    _print_scenes_table(scenes)
    _save_scenes(ws, scenes)
    typer.echo(f"\n✓ {len(scenes)} scenes detected")
    _next(f"sample-frames --film '{film}' --song '{song}'")


@app.command("sample-frames")
def sample_frames_cmd(
    film:     Annotated[str, typer.Option(help="Film name")],
    song:     Annotated[str, typer.Option(help="Song title")],
    n_frames: Annotated[int, typer.Option(help="Frames to extract per scene")] = N_FRAMES_PER_SCENE,
) -> None:
    """Extract representative frames from each scene → updates scenes.json."""
    from pipeline.frame_sampler import sample_all_scenes
    ws = _WS(film, song)
    video_path = _load_video_path(ws)
    scenes = _load_scenes(ws)
    scenes = sample_all_scenes(video_path, scenes, n_frames=n_frames, frames_dir=ws.frames_dir)
    _save_scenes(ws, scenes)
    typer.echo(f"\n✓ Sampled {n_frames} frames × {len(scenes)} scenes")
    _next(f"analyze-scenes --film '{film}' --song '{song}'")


@app.command("analyze-scenes")
def analyze_scenes_cmd(
    film:        Annotated[str,             typer.Option(help="Film name")],
    song:        Annotated[str,             typer.Option(help="Song title")],
    characters:  Annotated[Optional[str],   typer.Option(help="Comma-separated main character names")] = None,
    model:       Annotated[OpenRouterModel, typer.Option(help="Vision model to use")] = DEFAULT_ANALYZER_MODEL,
    max_batches: Annotated[Optional[int],   typer.Option(help="Stop after N batches (useful for testing)")] = None,
) -> None:
    """Describe each scene with a vision LLM and flag non-film scenes → updates scenes.json."""
    from pipeline.scene_analyzer import analyze_scenes
    ws = _WS(film, song)
    scenes = _load_scenes(ws)
    missing = [s for s in scenes if not s.frames]
    if missing:
        typer.echo(f"Error: {len(missing)} scenes have no frames. Run 'sample-frames' first.", err=True)
        raise typer.Exit(1)
    char_list = [c.strip() for c in characters.split(",")] if characters else None
    scenes = analyze_scenes(
        scenes, film_name=film, model=model.value,
        characters=char_list, max_batches=max_batches, cache_dir=ws.clip_cache_dir,
    )
    _save_scenes(ws, scenes)
    related   = sum(1 for s in scenes if s.is_film_related)
    aesthetic = sum(1 for s in scenes if s.is_aesthetic)
    usable    = sum(1 for s in scenes if s.is_usable)
    typer.echo(f"\n✓ {related}/{len(scenes)} film-related  |  {aesthetic}/{len(scenes)} aesthetic  |  {usable} usable for matching")
    _next(f"generate-plan --film '{film}' --song '{song}'")


@app.command("extract-lyrics")
def extract_lyrics_cmd(
    film:          Annotated[str,             typer.Option(help="Film name")],
    song:          Annotated[str,             typer.Option(help="Song title")],
    artist:        Annotated[Optional[str],   typer.Option(help="Artist name (improves LRCLIB lookup)")] = None,
    song_start:    Annotated[Optional[float], typer.Option(help="Window start (seconds)")] = None,
    song_end:      Annotated[Optional[float], typer.Option(help="Window end   (seconds)")] = None,
    refine_timing: Annotated[bool,            typer.Option(
        "--refine-timing/--no-refine-timing",
        help="Use Whisper word-level timestamps to show subtitles only while singing (default: on)",
    )] = True,
) -> None:
    """Fetch synced lyrics via LRCLIB (+ Whisper timing refinement) → saves state/lyrics.json."""
    from pipeline.lyrics_extractor import extract_lyrics
    ws = _WS(film, song)
    audio_path = _load_audio_path(ws)
    lyrics = extract_lyrics(
        audio_path, track=song, artist=artist,
        start_sec=song_start, end_sec=song_end,
        refine_timing=refine_timing,
    )
    if not lyrics:
        typer.echo("Error: no lyrics found.", err=True)
        raise typer.Exit(1)
    _save_lyrics(ws, lyrics)

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
        typer.echo(f"  … ({len(lyrics) - preview_n} more — see {ws.lyrics_readable_file.name})")

    total_dur = non_zero[-1].end_time if non_zero else 0.0
    typer.echo(
        f"\n✓ {len(lyrics)} lyric chunks  |  "
        f"{len(non_zero)} displayable  |  "
        f"song window: 0 – {total_dur:.1f}s"
    )
    typer.echo(f"  Full list: {ws.lyrics_readable_file}")

    # ── Warn if the existing plan won't match the new lyric count ─────────────
    if ws.plan_file.exists():
        try:
            plan_data = json.loads(ws.plan_file.read_text())
            plan_lyric_texts = {
                ll["text"].strip().lower()
                for seg in plan_data
                for ll in seg.get("lyric_lines", [])
            }
            if len(plan_lyric_texts) != len(lyrics):
                typer.echo(
                    f"\n⚠️  The existing plan has a different lyric count and will NOT sync "
                    f"correctly on render.\n"
                    f"   → Re-run: song-video-maker generate-plan --film '{film}' --song '{song}'"
                )
        except Exception:
            pass

    _next(f"generate-plan --film '{film}' --song '{song}'")


@app.command("generate-plan")
def generate_plan_cmd(
    film:       Annotated[str,             typer.Option(help="Film name")],
    song:       Annotated[str,             typer.Option(help="Song title")],
    model:      Annotated[OpenRouterModel, typer.Option(help="Text model to use")] = DEFAULT_MATCHER_MODEL,
    characters: Annotated[Optional[str],   typer.Option(help="Comma-separated main character names")] = None,
) -> None:
    """Ask an LLM to map scenes to lyrics → saves {slug}_plan.json + _readable.txt in workspace."""
    from pipeline.matcher import generate_plan
    ws = _WS(film, song)
    scenes = _load_scenes(ws)
    lyrics = _load_lyrics(ws)
    audio_path = _load_audio_path(ws)
    char_list = [c.strip() for c in characters.split(",")] if characters else None
    generate_plan(
        scenes=scenes,
        lyrics=lyrics,
        film_name=film,
        song_name=song,
        output_dir=ws.root,
        slug=ws.slug,
        model=model.value,
        characters=char_list,
        audio_path=audio_path,
        cache_dir=ws.plan_cache_dir,
    )
    typer.echo(f"\n✓ Plan: {ws.plan_readable_file}")
    _next(f"edit-plan --film '{film}' --song '{song}' --instruction '...'  OR  song-video-maker render --film '{film}' --song '{song}'")


@app.command("edit-plan")
def edit_plan_cmd(
    film:        Annotated[str,             typer.Option(help="Film name")],
    song:        Annotated[str,             typer.Option(help="Song title")],
    instruction: Annotated[str,             typer.Option(help="Natural-language editing instruction")],
    model:       Annotated[OpenRouterModel, typer.Option(help="OpenRouter model to use")] = DEFAULT_EDIT_MODEL,
    ollama:      Annotated[Optional[str],   typer.Option(help="Use local Ollama instead: provide model name (e.g. 'qwen2.5:14b')")] = None,
    ollama_url:  Annotated[str,             typer.Option(help="Ollama base URL")] = "http://localhost:11434",
    auto_render: Annotated[bool,            typer.Option("--auto-render/--no-auto-render", help="Re-render immediately after editing")] = False,
) -> None:
    """Edit the current plan with a natural-language instruction.

    Examples:
      --instruction "Clip 5 is too static, swap it to something with more movement"
      --instruction "Scene 22 is repeated too often, replace clips 3 and 7 with alternatives"
      --instruction "Fix lyric 2 — change 'there' to 'here'"
      --instruction "The intro needs more energy — use close-up shots for clips 1 and 2"

    The current plan is backed up before any changes are made.
    Lyric text corrections update state/lyrics.json directly.
    """
    from pipeline.plan_editor import edit_plan
    from pipeline.matcher import _build_timeline_slots, _write_readable_plan
    from pipeline.editor import _probe_duration as probe_dur

    ws = _WS(film, song)
    _require(ws.plan_file, "generate-plan", ws)

    scenes   = _load_scenes(ws)
    lyrics   = _load_lyrics(ws)
    segments = [MatchedSegment.from_dict(d) for d in json.loads(ws.plan_file.read_text())]

    # ── Backup current plan into plan_history/ ────────────────────────────────
    from pipeline.matcher import archive_existing_plan
    archived = archive_existing_plan(ws.root, ws.slug)
    if archived:
        typer.echo(f"[edit-plan] Archived → plan_history/{archived.name}")

    # ── Call the editor LLM ───────────────────────────────────────────────────
    edit_model  = ollama if ollama else model.value
    ollama_base = ollama_url if ollama else None

    new_segments, new_lyrics = edit_plan(
        segments=segments,
        scenes=scenes,
        lyrics=lyrics,
        instruction=instruction,
        film_name=film,
        song_name=song,
        model=edit_model,
        ollama_url=ollama_base,
    )

    # ── Save modified plan ────────────────────────────────────────────────────
    ws.plan_file.write_text(json.dumps([s.to_dict() for s in new_segments], indent=2))
    typer.echo(f"\n✓ Plan updated: {ws.plan_file}")

    # ── Save modified lyrics (if changed) ─────────────────────────────────────
    if new_lyrics != lyrics:
        _save_lyrics(ws, new_lyrics)
        typer.echo(f"✓ Lyrics updated: {ws.lyrics_file}")

    # ── Regenerate human-readable plan ───────────────────────────────────────
    try:
        audio_path     = _load_audio_path(ws)
        audio_duration = probe_dur(audio_path)
        slots          = _build_timeline_slots(new_lyrics, audio_duration)
        scene_by_index = {s.index: s for s in scenes}
        _write_readable_plan(new_segments, scene_by_index, slots, film, song, ws.plan_readable_file)
        typer.echo(f"✓ Readable plan: {ws.plan_readable_file}")
    except Exception as e:
        typer.echo(f"  (readable plan not regenerated: {e})")

    # ── Optional auto-render ──────────────────────────────────────────────────
    if auto_render:
        from pipeline.editor import render_video
        typer.echo("\n── Auto-rendering ────────────────────────────────────────")
        cfg          = json.loads(ws.config_file.read_text()) if ws.config_file.exists() else {}
        video_path   = _load_video_path(ws)
        audio_path   = _load_audio_path(ws)
        fresh_lyrics = _load_lyrics_safe(ws)
        render_video(
            new_segments, scenes, video_path, audio_path, ws.reel_file,
            artist=cfg.get("artist", ""),
            song_title=ws.song,
            film_name=ws.film,
            director=cfg.get("director", ""),
            lyrics=fresh_lyrics or None,
        )
        typer.echo(f"\n✓ Reel: {ws.reel_file}")
    else:
        _next(f"render --film '{film}' --song '{song}'")


@app.command()
def render(
    film:              Annotated[str,            typer.Option(help="Film name")],
    song:              Annotated[str,            typer.Option(help="Song title")],
    director:          Annotated[Optional[str],  typer.Option(help="Director name for credit insert")] = None,
    subtitle_font:     Annotated[Optional[str],  typer.Option(help="Subtitle font name, e.g. 'Helvetica', 'Futura'")] = None,
    subtitle_fontsize: Annotated[Optional[int],  typer.Option(help="Subtitle font size (default: 56)")] = None,
    subtitle_italic:   Annotated[bool, typer.Option("--subtitle-italic/--no-subtitle-italic", help="Render subtitles in italic")] = False,
    publish:           Annotated[bool, typer.Option("--publish", help="Publish to Instagram after rendering")] = False,
    caption:           Annotated[Optional[str],  typer.Option(help="Caption for Instagram post")] = None,
) -> None:
    """Assemble the final 9:16 Reel from the plan → {slug}_reel.mp4 in workspace."""
    from pipeline.editor import render_video
    from config import SUBTITLE_FONTS
    ws = _WS(film, song)
    if director:
        _update_config(ws, director=director)
    cfg = json.loads(ws.config_file.read_text()) if ws.config_file.exists() else {}

    # Resolve subtitle font name → absolute path
    font_path = ""
    if subtitle_font:
        font_path = SUBTITLE_FONTS.get(subtitle_font, "")
        if not font_path:
            from pathlib import Path as _Path
            if _Path(subtitle_font).exists():
                font_path = subtitle_font
            else:
                typer.echo(f"⚠ Font '{subtitle_font}' not found — using default", err=True)

    video_path = _load_video_path(ws)
    audio_path = _load_audio_path(ws)
    scenes     = _load_scenes(ws)
    fresh_lyrics = _load_lyrics_safe(ws)
    _require(ws.plan_file, "generate-plan", ws)
    segments = [MatchedSegment.from_dict(d) for d in json.loads(ws.plan_file.read_text())]
    if fresh_lyrics:
        typer.echo(f"[render] Using {len(fresh_lyrics)} lines from lyrics.json for subtitle timing")
    render_video(
        segments, scenes, video_path, audio_path, ws.reel_file,
        artist=cfg.get("artist", ""),
        song_title=song,
        film_name=film,
        director=cfg.get("director", director or ""),
        subtitle_font=font_path,
        subtitle_fontsize=subtitle_fontsize or 0,
        subtitle_italic=subtitle_italic,
        manual_crop=cfg.get("manual_crop") or None,
        lyrics=fresh_lyrics or None,
    )
    typer.echo(f"\n✓ Reel: {ws.reel_file}")

    if publish:
        from pipeline.publisher import publish_to_instagram
        default_caption = f"'{song}' by {film} (AI generated reel) #aesthetic #reels"
        publish_to_instagram(ws.reel_file, caption or default_caption)


@app.command()
def publish(
    film: Annotated[str, typer.Option(help="Film name")],
    song: Annotated[str, typer.Option(help="Song title")],
    caption: Annotated[Optional[str], typer.Option(help="Caption for Instagram post")] = None,
) -> None:
    """Publish an existing Reel to Instagram."""
    from pipeline.publisher import publish_to_instagram
    ws = _WS(film, song)
    _require(ws.reel_file, "render", ws)
    
    # Try to get artist from config for a better default caption
    artist = ""
    if ws.config_file.exists():
        try:
            cfg = json.loads(ws.config_file.read_text())
            artist = f" by {cfg.get('artist', '')}" if cfg.get('artist') else ""
        except:
            pass
            
    default_caption = f"{song}{artist} — a {film} aesthetic reel. #aesthetic #reels #ai"
    publish_to_instagram(ws.reel_file, caption or default_caption)


@app.command()
def status(
    film: Annotated[Optional[str], typer.Option(help="Film name (omit to list all workspaces)")] = None,
    song: Annotated[Optional[str], typer.Option(help="Song title (omit to list all workspaces)")] = None,
) -> None:
    """Show pipeline status. Omit --film/--song to list all workspaces."""
    if film and song:
        _show_workspace_status(_WS(film, song))
    else:
        _list_workspaces()


# ── Status helpers ────────────────────────────────────────────────────────────

def _list_workspaces() -> None:
    if not WORKSPACES_DIR.exists():
        typer.echo("No workspaces yet. Run a pipeline command to create one.")
        return
    dirs = sorted(d for d in WORKSPACES_DIR.iterdir() if d.is_dir())
    if not dirs:
        typer.echo("No workspaces yet. Run a pipeline command to create one.")
        return

    typer.echo(f"\n── Workspaces ({len(dirs)}) ─────────────────────────────────────────────")
    for ws_dir in dirs:
        state_dir = ws_dir / "state"
        parts = []
        if (state_dir / "video_path.txt").exists(): parts.append("video")
        if (state_dir / "audio_path.txt").exists(): parts.append("audio")
        if (state_dir / "scenes.json").exists():    parts.append("scenes")
        if (state_dir / "lyrics.json").exists():    parts.append("lyrics")
        if next(ws_dir.glob("*_plan.json"), None):  parts.append("plan")
        if next(ws_dir.glob("*_reel.mp4"), None):   parts.append("reel ✓")
        pipeline = " → ".join(parts) if parts else "(empty)"
        typer.echo(f"  {ws_dir.name}")
        typer.echo(f"    {pipeline}")

    typer.echo(f"\nFor details: song-video-maker status --film '<film>' --song '<song>'")


def _show_workspace_status(ws: _WS) -> None:
    def _row(label: str, path: Path, detail: str = "") -> None:
        mark = typer.style("✓", fg=typer.colors.GREEN) if path.exists() else typer.style("✗", fg=typer.colors.RED)
        size = f"({path.stat().st_size // 1024}KB)" if path.exists() else ""
        typer.echo(f"  {mark}  {label:<40} {size:<10} {detail}")

    typer.echo(f"\n── Workspace: {ws.root}")
    typer.echo(f"   Slug: {ws.slug}")
    if ws.config_file.exists():
        cfg = json.loads(ws.config_file.read_text())
        typer.echo(f"   Created: {cfg.get('created_at', '?')}  |  Artist: {cfg.get('artist', '?')}")
    typer.echo()

    typer.echo("── Downloads ─────────────────────────────────────────────────")
    video_display = Path(ws.video_path_file.read_text().strip()) if ws.video_path_file.exists() else ws.root / "video.mp4"
    _row(video_display.name, video_display)
    audio_display = Path(ws.audio_path_file.read_text().strip()) if ws.audio_path_file.exists() else ws.root / "audio.mp3"
    _row(audio_display.name, audio_display)

    typer.echo("\n── State ─────────────────────────────────────────────────────")
    if ws.scenes_file.exists():
        scenes    = [Scene.model_validate(d) for d in json.loads(ws.scenes_file.read_text())]
        related   = sum(1 for s in scenes if s.is_film_related)
        n_frames  = sum(1 for s in scenes if s.frames)
        n_desc    = sum(1 for s in scenes if s.description)
        aesthetic = sum(1 for s in scenes if s.is_aesthetic)
        usable    = sum(1 for s in scenes if s.is_usable)
        detail = f"{len(scenes)} scenes  {n_frames} w/frames  {n_desc} analyzed  {related} film-related  {aesthetic} aesthetic  {usable} usable"
        _row("state/scenes.json", ws.scenes_file, detail)
    else:
        _row("state/scenes.json", ws.scenes_file)

    if ws.lyrics_file.exists():
        lyrics  = [LyricLine.model_validate(d) for d in json.loads(ws.lyrics_file.read_text())]
        preview = (lyrics[0].text[:40] + "...") if lyrics else ""
        detail  = f'{len(lyrics)} lines  "{preview}"'
        _row("state/lyrics.json", ws.lyrics_file, detail)
    else:
        _row("state/lyrics.json", ws.lyrics_file)

    typer.echo("\n── Outputs ───────────────────────────────────────────────────")
    _row(ws.plan_file.name,          ws.plan_file)
    _row(ws.plan_readable_file.name, ws.plan_readable_file)
    _row(ws.reel_file.name,          ws.reel_file)
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
