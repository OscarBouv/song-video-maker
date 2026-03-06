"""Transcribe song lyrics with timestamps using faster-whisper."""
from pathlib import Path

from faster_whisper import WhisperModel

from config import WHISPER_MODEL
from models import LyricLine

# Silence/pause threshold to break words into lines (seconds)
LINE_BREAK_GAP = 0.6
# Maximum words per line before forcing a break
MAX_WORDS_PER_LINE = 8


def extract_lyrics(
    audio_path: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
    model_size: str = WHISPER_MODEL,
) -> list[LyricLine]:
    """Transcribe audio and return LyricLine list with timestamps.

    If start_sec/end_sec are provided, timestamps are adjusted to be relative
    to start_sec (so the first lyric starts near 0.0).
    """
    print(f"[lyrics_extractor] Loading Whisper model '{model_size}'...")
    model = WhisperModel(model_size, device="auto", compute_type="default")

    print(f"[lyrics_extractor] Transcribing {audio_path.name}...")
    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=None,  # auto-detect
    )

    offset = start_sec if start_sec is not None else 0.0

    # Collect all words with timestamps
    all_words: list[tuple[float, float, str]] = []
    for segment in segments:
        if segment.words is None:
            continue
        for word in segment.words:
            word_start = word.start - offset
            word_end = word.end - offset

            # Skip words outside our requested window
            if end_sec is not None and word_start > (end_sec - offset):
                break
            if word_start < 0:
                continue

            all_words.append((word_start, word_end, word.word.strip()))

    if not all_words:
        print("[lyrics_extractor] Warning: no words found in transcription")
        return []

    # Group words into lines by gap or max word count
    lines: list[LyricLine] = []
    current_words: list[tuple[float, float, str]] = []

    for i, (w_start, w_end, w_text) in enumerate(all_words):
        if current_words:
            gap = w_start - current_words[-1][1]
            if gap > LINE_BREAK_GAP or len(current_words) >= MAX_WORDS_PER_LINE:
                lines.append(_make_line(current_words))
                current_words = []

        current_words.append((w_start, w_end, w_text))

    if current_words:
        lines.append(_make_line(current_words))

    print(f"[lyrics_extractor] Extracted {len(lines)} lyric lines")
    for line in lines:
        print(f"  {line}")
    return lines


def _make_line(words: list[tuple[float, float, str]]) -> LyricLine:
    text = " ".join(w[2] for w in words)
    return LyricLine(
        text=text,
        start_time=words[0][0],
        end_time=words[-1][1],
    )
