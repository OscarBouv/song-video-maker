"""Download video and audio from YouTube using yt-dlp."""
import re
import subprocess
from pathlib import Path

import yt_dlp

from config import TEMP_DIR, FFMPEG_BIN


def _slugify(*parts: str | None) -> str:
    text = "_".join(p for p in parts if p)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def download_video(
    url: str,
    output_dir: Path = TEMP_DIR,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> Path:
    """Download up to 1080p video+audio to output_dir, optionally trimmed. Returns path to file.

    Caps at 1080p — higher resolutions waste bandwidth since the output is 1080×1920.
    Accepts any container (webm/mp4) since everything is re-encoded by the editor.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "video.%(ext)s")

    ydl_opts = {
        # Prefer ≤1080p: no benefit from 4K source when output is 1080×1920.
        # Accept any container — the editor re-encodes everything anyway.
        "format": (
            "bestvideo[height<=1080]+bestaudio[ext=m4a]"
            "/bestvideo[height<=1080]+bestaudio"
            "/bestvideo+bestaudio"
            "/best[height<=1080]/best"
        ),
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        path = Path(filename).with_suffix(".mp4")
        if not path.exists():
            candidates = list(output_dir.glob("video.*"))
            if not candidates:
                raise FileNotFoundError(f"Downloaded video not found in {output_dir}")
            path = candidates[0]

    if start_sec is not None or end_sec is not None:
        suffix = f"_{int(start_sec or 0)}_{int(end_sec or 0)}"
        trimmed = path.parent / f"video{suffix}.mp4"
        # Re-encode trim for frame-accurate start point (stream copy can miss the first keyframe)
        cmd = [FFMPEG_BIN, "-y"]
        if start_sec is not None:
            cmd += ["-ss", str(start_sec)]
        cmd += ["-i", str(path)]
        if end_sec is not None:
            # -to is relative to -ss when -ss is before -i, so use duration instead
            duration = (end_sec - (start_sec or 0))
            cmd += ["-t", str(duration)]
        cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(trimmed)]
        subprocess.run(cmd, check=True, capture_output=True)
        path = trimmed

    print(f"[downloader] Video saved: {path}")
    return path


def download_audio(
    url: str,
    output_dir: Path = TEMP_DIR,
    start_sec: float | None = None,
    end_sec: float | None = None,
    track: str | None = None,
    artist: str | None = None,
) -> Path:
    """Download audio as high-quality MP3 to output_dir, optionally trimmed.
    Returns path to the audio file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(artist, track) or "audio"
    output_template = str(output_dir / f"{slug}.%(ext)s")

    postprocessors = [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "320",  # was 192 — max MP3 quality for the source track
        }
    ]

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",  # prefer M4A source for AAC→MP3 quality
        "outtmpl": output_template,
        "postprocessors": postprocessors,
        "noplaylist": True,
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    path = output_dir / f"{slug}.mp3"
    if not path.exists():
        candidates = list(output_dir.glob(f"{slug}.*"))
        if not candidates:
            raise FileNotFoundError(f"Downloaded audio not found in {output_dir}")
        path = candidates[0]

    # If trimming needed but couldn't be done via postprocessor, trim with ffmpeg
    if (start_sec is not None or end_sec is not None) and path.exists():
        path = _trim_audio(path, start_sec, end_sec, slug)

    print(f"[downloader] Audio saved: {path}")
    return path


def _trim_audio(
    audio_path: Path, start_sec: float | None, end_sec: float | None, slug: str = "audio"
) -> Path:
    """Trim audio with ffmpeg, re-encoding for sample-accurate start point."""
    suffix = f"_{int(start_sec or 0)}_{int(end_sec or 0)}"
    trimmed = audio_path.parent / f"{slug}{suffix}.mp3"
    # Place -ss before -i for fast seek, then re-encode so the start is sample-accurate
    cmd = [FFMPEG_BIN, "-y"]
    if start_sec is not None:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", str(audio_path)]
    if end_sec is not None:
        duration = end_sec - (start_sec or 0)
        cmd += ["-t", str(duration)]
    cmd += ["-c:a", "libmp3lame", "-b:a", "320k", "-q:a", "0", str(trimmed)]

    subprocess.run(cmd, check=True, capture_output=True)
    return trimmed
