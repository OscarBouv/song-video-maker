"""Extract synced lyrics for a song.

Strategy (in order):
  1. LRCLIB  — free public API, returns pre-timed LRC lyrics for most popular songs.
               No API key, no cost, stdlib HTTP only.
  2. Whisper — local faster-whisper transcription as fallback for obscure tracks.
               Only loaded if LRCLIB returns nothing.

Timing refinement (always on by default):
  After fetching LRCLIB lyrics (which only give line *start* times, with end = next
  line's start), we run faster-whisper with word-level timestamps to find the exact
  window during which each line is actually being *sung*. This ensures subtitles
  appear only while the singer is pronouncing the words — not during silences.
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

# Whisper fallback / refinement config
_LINE_BREAK_GAP = 0.6   # seconds of silence to split a new line (fallback mode)
_MAX_WORDS_PER_LINE = 8
_REFINE_MAX_GAP      = 0.5  # max silence between words to keep them in the same cluster
_REFINE_MAX_FORWARD  = 4.0  # max seconds *after* LRCLIB expected start still considered a valid match

_CACHE_DIR = Path(__file__).parent.parent / "temp" / "cache"


# ── Public entry point ────────────────────────────────────────────────────────

def extract_lyrics(
    audio_path: Path,
    track: str,
    artist: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    refine_timing: bool = True,
) -> list[LyricLine]:
    """Return a LyricLine list for the song, timestamps relative to start_sec.

    Tries LRCLIB first (free, synced). When successful and refine_timing=True,
    runs faster-whisper word-level alignment to tighten each subtitle to the
    exact pronunciation window (no subtitle during inter-line silences).

    Falls back to pure Whisper transcription when LRCLIB returns nothing.
    """
    lines = _fetch_lrclib(track, artist)

    if lines:
        lines = _apply_window(lines, start_sec, end_sec)
        if refine_timing:
            print(
                f"[lyrics_extractor] Refining {len(lines)} LRCLIB timestamps "
                f"with Whisper word-level alignment..."
            )
            lines = _refine_timestamps_with_whisper(lines, audio_path)
        else:
            print(f"[lyrics_extractor] {len(lines)} lines from LRCLIB (timing not refined)")
            for ll in lines:
                print(f"  {ll}")
        lines = _split_into_chunks(lines)
        n_chunks = len(lines)
        print(f"[lyrics_extractor] {n_chunks} subtitle chunks after splitting (≤{_MAX_SUBTITLE_WORDS} words each)")
        return lines

    print("[lyrics_extractor] LRCLIB miss — falling back to Whisper transcription")
    lines = _extract_with_whisper(audio_path, start_sec, end_sec)
    lines = _split_into_chunks(lines)
    print(f"[lyrics_extractor] {len(lines)} subtitle chunks after splitting (≤{_MAX_SUBTITLE_WORDS} words each)")
    return lines


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
    These are approximate — use _refine_timestamps_with_whisper() for precision.
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


# ── Whisper timing refinement ─────────────────────────────────────────────────

def _get_whisper_words(
    audio_path: Path,
    model_size: str = WHISPER_MODEL,
) -> list[tuple[float, float, str]]:
    """Return word-level timestamps from faster-whisper, with disk caching.

    The audio_path is transcribed as-is; timestamps are relative to the start
    of the file (which is already trimmed to the song window by the downloader).

    Returns list of (start_sec, end_sec, word_text).
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"whisper_words_{audio_path.stem}.json"

    if cache_file.exists():
        print(f"[lyrics_extractor] Whisper cache hit → {cache_file.name}")
        with open(cache_file) as f:
            raw = json.load(f)
        return [(float(w[0]), float(w[1]), str(w[2])) for w in raw]

    print(f"[lyrics_extractor] Running Whisper '{model_size}' on {audio_path.name}...")
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="auto", compute_type="default")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)

    words: list[tuple[float, float, str]] = []
    for segment in segments:
        if not segment.words:
            continue
        for word in segment.words:
            words.append((round(word.start, 3), round(word.end, 3), word.word.strip()))

    with open(cache_file, "w") as f:
        json.dump(words, f)
    print(f"[lyrics_extractor] Whisper found {len(words)} words → cached to {cache_file.name}")
    return words


def _find_pronunciation_window(
    search_from: float,
    search_end: float,
    words: list[tuple[float, float, str]],
    max_gap: float = _REFINE_MAX_GAP,
) -> tuple[float, float] | None:
    """Find the first contiguous vocal cluster in [search_from, search_end).

    Collects words that start within the window, then walks forward while
    the silence gap between consecutive words is ≤ max_gap.

    Returns (cluster_start, cluster_end) or None if no words found.
    """
    window = [(ws, we, t) for ws, we, t in words if ws >= search_from and ws < search_end]
    if not window:
        return None

    cluster = [window[0]]
    for ws, we, t in window[1:]:
        if ws - cluster[-1][1] > max_gap:
            break
        cluster.append((ws, we, t))

    return round(cluster[0][0], 3), round(cluster[-1][1], 3)


def _refine_timestamps_with_whisper(
    lines: list[LyricLine],
    audio_path: Path,
) -> list[LyricLine]:
    """Replace LRCLIB line timestamps with precise Whisper word-level timing.

    Strategy:
    - We search for each line's vocals starting from where the *previous* line ended,
      so there can be no overlaps between consecutive lines.
    - Each line's end is hard-clipped to just before the next LRCLIB line's start,
      splitting continuous singing across two lines at the correct boundary.
    - Lines where no Whisper words are found (before audio start, instrumentals)
      get zero-duration timestamps so they are never displayed.
    """
    words = _get_whisper_words(audio_path)
    if not words:
        print("[lyrics_extractor] Whisper returned no words — keeping LRCLIB timestamps")
        for ll in lines:
            print(f"  {ll}")
        return lines

    refined: list[LyricLine] = []
    prev_end = 0.0  # end timestamp of the last successfully placed subtitle

    for i, line in enumerate(lines):
        # Hard upper bound: just before the next LRCLIB line's expected start
        next_start = lines[i + 1].start_time if i + 1 < len(lines) else line.start_time + 10.0
        search_end = next_start - 0.1

        result = _find_pronunciation_window(prev_end, search_end, words)

        # Sanity check: reject if the first word found is too far *after* the expected
        # LRCLIB time — indicates we're matching the wrong line (e.g., a line that starts
        # before the audio window, where Whisper picks up the next line's words instead).
        if result and result[0] > line.start_time + _REFINE_MAX_FORWARD:
            result = None

        if result:
            actual_start = result[0]
            actual_end   = min(result[1], search_end)  # hard clip at LRCLIB boundary
            prev_end = actual_end
            refined.append(LyricLine(
                text=line.text,
                start_time=actual_start,
                end_time=actual_end,
            ))
            dur = actual_end - actual_start
            print(f"  {line.start_time:+6.2f}s → [{actual_start:.2f}–{actual_end:.2f}s, {dur:.1f}s]  {line.text[:55]}")
        else:
            # No singing found — advance prev_end past this line so next line searches
            # from the right place, then emit a zero-duration entry (never displayed).
            prev_end = max(prev_end, max(0.0, line.start_time))
            fallback_t = max(0.0, line.start_time)
            refined.append(LyricLine(
                text=line.text,
                start_time=fallback_t,
                end_time=fallback_t,  # zero-duration → drawtext enable never fires
            ))
            print(f"  {line.start_time:+6.2f}s → [skipped — no matching audio]  {line.text[:55]}")

    return refined


# ── Whisper fallback (when LRCLIB returns nothing) ────────────────────────────

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
    for ll in lines:
        print(f"  {ll}")
    return lines


def _words_to_line(words: list[tuple[float, float, str]]) -> LyricLine:
    return LyricLine(
        text=" ".join(w[2] for w in words),
        start_time=words[0][0],
        end_time=words[-1][1],
    )


# ── Subtitle chunking ─────────────────────────────────────────────────────────

_MAX_SUBTITLE_WORDS = 6   # lines longer than this are split into ≤ N-word chunks


def _split_into_chunks(
    lines: list[LyricLine],
    max_words: int = _MAX_SUBTITLE_WORDS,
) -> list[LyricLine]:
    """Split lyric lines that are longer than *max_words* into proportionally-timed chunks.

    Example (max_words=6):
      "Can I waste all your time here on the sidewalk"  (9 words, 5.88–13.02 s)
        → "Can I waste all your time"  (6 words, 5.88–10.64 s)
        + "here on the sidewalk"       (3 words, 10.64–13.02 s)

    Time is distributed proportionally by word count.
    Zero-duration lines (start_time == end_time, i.e. "never-display" markers)
    are passed through unchanged so they remain invisible in the final render.
    """
    result: list[LyricLine] = []
    for line in lines:
        words = line.text.split()
        # Pass through: short enough, or zero-duration never-display marker
        if len(words) <= max_words or line.start_time >= line.end_time:
            result.append(line)
            continue

        total_words = len(words)
        duration    = line.end_time - line.start_time
        t = line.start_time

        pos = 0
        while pos < total_words:
            chunk = words[pos : pos + max_words]
            chunk_word_count = len(chunk)
            chunk_dur = duration * chunk_word_count / total_words
            chunk_end = round(t + chunk_dur, 3)

            result.append(LyricLine(
                text=(" ".join(chunk)),
                start_time=round(t, 3),
                end_time=chunk_end,
            ))
            t = chunk_end
            pos += max_words

    return result
