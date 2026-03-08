"""Extract synced lyrics for a song.

Strategy (in order):
  1. LRCLIB  — free public API, returns pre-timed LRC lyrics for most popular songs.
               No API key, no cost, stdlib HTTP only.
  2. Whisper — local faster-whisper transcription as fallback for obscure tracks.
               Only loaded if LRCLIB returns nothing.
"""
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from config import WHISPER_MODEL
from models import LyricLine

LRCLIB_API = "https://lrclib.net/api"
# Matches: [MM:SS.xx] lyric text
_LRC_RE = re.compile(r"\[(\d+):(\d+\.\d+)\]\s*(.*)")

# Whisper fallback config
_LINE_BREAK_GAP = 0.6   # seconds of silence to split a new line
_MAX_WORDS_PER_LINE = 8


# ── Public entry point ────────────────────────────────────────────────────────

def extract_lyrics(
    audio_path: Path,
    track: str,
    artist: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> list[LyricLine]:
    """Return a LyricLine list for the song, timestamps relative to start_sec.

    Tries LRCLIB first; falls back to local Whisper transcription.
    """
    lines = _fetch_lrclib(track, artist)

    if lines:
        lines = _apply_window(lines, start_sec, end_sec)
        print(f"[lyrics_extractor] {len(lines)} lines from LRCLIB")
        for l in lines:
            print(f"  {l}")
        return lines

    print("[lyrics_extractor] LRCLIB miss — falling back to Whisper transcription")
    return _extract_with_whisper(audio_path, start_sec, end_sec)


# ── LRCLIB ────────────────────────────────────────────────────────────────────

def _fetch_lrclib(track: str, artist: str | None) -> list[LyricLine] | None:
    """Query LRCLIB for synced lyrics. Returns None if nothing found."""
    params: dict[str, str] = {"track_name": track}
    if artist:
        params["artist_name"] = artist

    url = f"{LRCLIB_API}/search?" + urllib.parse.urlencode(params)
    print(f"[lyrics_extractor] Querying LRCLIB: {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "song-video-maker/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            results = json.loads(resp.read())
    except Exception as e:
        print(f"[lyrics_extractor] LRCLIB request failed: {e}")
        return None

    for result in results:
        synced = result.get("syncedLyrics") or ""
        if not synced.strip():
            continue
        lines = _parse_lrc(synced)
        if lines:
            print(f"[lyrics_extractor] LRCLIB match: '{result.get('artistName')} — {result.get('trackName')}'")
            return lines

    return None


def _parse_lrc(lrc: str) -> list[LyricLine]:
    """Parse LRC format into LyricLine objects.

    LRC lines look like: [01:23.45] Some lyric text
    End time of each line = start time of the next; last line gets +5s.
    """
    raw: list[tuple[float, str]] = []
    for line in lrc.splitlines():
        m = _LRC_RE.match(line.strip())
        if not m:
            continue
        text = m.group(3).strip()
        if not text:
            continue  # skip empty / instrumental markers
        t = int(m.group(1)) * 60 + float(m.group(2))
        raw.append((t, text))

    lines: list[LyricLine] = []
    for i, (start, text) in enumerate(raw):
        end = raw[i + 1][0] if i + 1 < len(raw) else start + 5.0
        lines.append(LyricLine(text=text, start_time=start, end_time=end))
    return lines


def _apply_window(
    lines: list[LyricLine],
    start_sec: float | None,
    end_sec: float | None,
) -> list[LyricLine]:
    """Filter lines to [start_sec, end_sec] window and shift timestamps to be 0-based."""
    offset = start_sec or 0.0

    result = []
    for l in lines:
        if start_sec is not None and l.end_time <= start_sec:
            continue
        if end_sec is not None and l.start_time >= end_sec:
            break
        result.append(
            LyricLine(
                text=l.text,
                start_time=round(l.start_time - offset, 3),
                end_time=round(l.end_time - offset, 3),
            )
        )
    return result


# ── Whisper fallback ──────────────────────────────────────────────────────────

def _extract_with_whisper(
    audio_path: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
    model_size: str = WHISPER_MODEL,
) -> list[LyricLine]:
    """Transcribe audio with faster-whisper and group words into lines."""
    from faster_whisper import WhisperModel

    print(f"[lyrics_extractor] Loading Whisper model '{model_size}'...")
    model = WhisperModel(model_size, device="auto", compute_type="default")

    print(f"[lyrics_extractor] Transcribing {audio_path.name}...")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True, language=None)

    offset = start_sec or 0.0
    all_words: list[tuple[float, float, str]] = []

    for segment in segments:
        if not segment.words:
            continue
        for word in segment.words:
            ws = word.start - offset
            we = word.end - offset
            if end_sec is not None and ws > (end_sec - offset):
                break
            if ws < 0:
                continue
            all_words.append((ws, we, word.word.strip()))

    if not all_words:
        print("[lyrics_extractor] Warning: Whisper found no words")
        return []

    lines: list[LyricLine] = []
    current: list[tuple[float, float, str]] = []

    for ws, we, text in all_words:
        if current:
            gap = ws - current[-1][1]
            if gap > _LINE_BREAK_GAP or len(current) >= _MAX_WORDS_PER_LINE:
                lines.append(_words_to_line(current))
                current = []
        current.append((ws, we, text))

    if current:
        lines.append(_words_to_line(current))

    print(f"[lyrics_extractor] Whisper extracted {len(lines)} lines")
    for l in lines:
        print(f"  {l}")
    return lines


def _words_to_line(words: list[tuple[float, float, str]]) -> LyricLine:
    return LyricLine(
        text=" ".join(w[2] for w in words),
        start_time=words[0][0],
        end_time=words[-1][1],
    )
