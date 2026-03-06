"""Download video and audio from YouTube using yt-dlp."""
from pathlib import Path

import yt_dlp

from config import TEMP_DIR


def download_video(url: str, output_dir: Path = TEMP_DIR) -> Path:
    """Download best quality video+audio to output_dir. Returns path to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "video.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # Ensure .mp4 extension after merge
        path = Path(filename).with_suffix(".mp4")
        if not path.exists():
            # fallback: find any downloaded mp4
            candidates = list(output_dir.glob("video.*"))
            if not candidates:
                raise FileNotFoundError(f"Downloaded video not found in {output_dir}")
            path = candidates[0]

    print(f"[downloader] Video saved: {path}")
    return path


def download_audio(
    url: str,
    output_dir: Path = TEMP_DIR,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> Path:
    """Download audio as MP3 to output_dir, optionally trimmed to [start_sec, end_sec].
    Returns path to the audio file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "audio.%(ext)s")

    postprocessors = [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ]

    # Build ffmpeg trim args if timecodes given
    postprocessor_args = []
    if start_sec is not None or end_sec is not None:
        if start_sec is not None:
            postprocessor_args += ["-ss", str(start_sec)]
        if end_sec is not None:
            postprocessor_args += ["-to", str(end_sec)]

    if postprocessor_args:
        postprocessors.append(
            {
                "key": "FFmpegPostProcessor",
                "args": postprocessor_args,
            }
        )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": postprocessors,
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    path = output_dir / "audio.mp3"
    if not path.exists():
        candidates = list(output_dir.glob("audio.*"))
        if not candidates:
            raise FileNotFoundError(f"Downloaded audio not found in {output_dir}")
        path = candidates[0]

    # If trimming needed but couldn't be done via postprocessor, trim with ffmpeg
    if (start_sec is not None or end_sec is not None) and path.exists():
        path = _trim_audio(path, start_sec, end_sec)

    print(f"[downloader] Audio saved: {path}")
    return path


def _trim_audio(
    audio_path: Path, start_sec: float | None, end_sec: float | None
) -> Path:
    """Trim audio using ffmpeg subprocess as fallback."""
    import subprocess

    trimmed = audio_path.parent / "audio_trimmed.mp3"
    cmd = ["ffmpeg", "-y", "-i", str(audio_path)]
    if start_sec is not None:
        cmd += ["-ss", str(start_sec)]
    if end_sec is not None:
        cmd += ["-to", str(end_sec)]
    cmd += ["-c", "copy", str(trimmed)]

    subprocess.run(cmd, check=True, capture_output=True)
    return trimmed
