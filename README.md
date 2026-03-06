# song-video-maker

Automatically generate aesthetic 9:16 Instagram Reels from movie clip compilations and songs — with AI-assisted scene selection, thematic scene-to-lyric matching, and yellow lyric subtitles.

Built entirely in Python, driven by AI at every analysis step.

---

## Demo idea

Input: a YouTube compilation of *Licorice Pizza* clips + *Call it Fate, Call it Karma* by The Strokes.
Output: a 90-second vertical Reel where scenes are cut to match the song's lyric timing and mood, subtitled in yellow.

---

## Architecture

The pipeline is split into **6 sequential steps**, each in its own module under `pipeline/`. A deliberate human review checkpoint sits between the AI planning step and the final render.

```
YouTube URL (clips)   ──► downloader ──► scene_detector ──► frame_sampler ──► scene_analyzer ──►┐
                                                                                                  ├──► matcher ──► [REVIEW] ──► editor ──► Reel.mp4
YouTube URL (song)    ──► downloader ──► lyrics_extractor ────────────────────────────────────►──┘
```

### Step-by-step

#### 1. `pipeline/downloader.py` — Download
Uses **yt-dlp** (Python API, no subprocess) to download:
- The video compilation as `temp/video.mp4`
- The song as `temp/audio.mp3`, with optional ffmpeg trimming to `[--song-start, --song-end]`

#### 2. `pipeline/scene_detector.py` — Scene cut detection
Uses **PySceneDetect** (`ContentDetector`) to find every hard cut in the compilation video.
Returns a list of `Scene` objects with `start_time` / `end_time` in seconds.
Scenes shorter than `MIN_SCENE_DURATION` (1.5s) are discarded.

#### 3. `pipeline/frame_sampler.py` — Frame extraction
Uses **OpenCV** to extract `N_FRAMES_PER_SCENE` (default: 3) evenly-spaced frames from each scene.
Frames are saved as JPEGs to `temp/frames/scene_{index}/` and stored on the `Scene` model.

#### 4. `pipeline/scene_analyzer.py` — AI scene understanding *(Claude vision)*
Sends scene frames in batches to **Claude** (vision model) via the Anthropic SDK.
For each scene, Claude returns:
- `description`: one-sentence visual summary
- `is_film_related`: whether the scene appears to be from the target film
- `confidence`: 0–1 score

Scenes from other films in the compilation are flagged (`is_film_related=False`) and excluded from the matching step.

#### 5. `pipeline/lyrics_extractor.py` — Lyrics transcription
Uses **faster-whisper** (local, no API cost) to transcribe the song audio with **word-level timestamps**.
Words are grouped into `LyricLine` objects by silence gaps (> 0.6s) or line length (> 8 words).
Timestamps are adjusted relative to `--song-start` if a window was specified.

#### 6. `pipeline/matcher.py` — AI scene-to-lyric plan *(LLM)*
Sends Claude (or an OpenRouter model) the full list of scene descriptions and lyric lines.
The model returns a JSON mapping of which scene should play during which lyrics, with a one-sentence rationale per match.

The pipeline **pauses here** and saves two files:
- `outputs/{slug}_plan.json` — machine-readable, editable
- `outputs/{slug}_plan_readable.txt` — human-friendly preview

You review and optionally edit the plan, then re-run with `--render-only` to render.

#### 7. `pipeline/editor.py` — Video assembly *(MoviePy + ffmpeg)*
Reads the plan, then for each segment:
1. Cuts the clip from the source video
2. Center-crops to **1080×1920** (9:16 vertical for Reels)
3. Overlays yellow `TextClip` subtitles (Arial Bold, stroke outlined) timed to each lyric line
4. Concatenates all clips and mixes in the song audio

Output: `outputs/{slug}_reel.mp4` at 30fps, max 90 seconds.

---

## Data models (`models/`)

All models use **Pydantic `BaseModel`** for runtime validation, clean JSON serialization (`model_dump`) and deserialization (`model_validate`).

| Model | Fields | Purpose |
|---|---|---|
| `Scene` | `index`, `start_time`, `end_time`, `frames`, `description`, `is_film_related`, `confidence` | One detected scene from the source video |
| `LyricLine` | `text`, `start_time`, `end_time` | One line of lyrics with song-relative timestamps |
| `MatchedSegment` | `scene_index`, `lyric_lines`, `scene_trim_start`, `scene_trim_end` | One entry in the final plan: a scene paired with its lyrics |

---

## LLM providers

The **scene analyzer** always uses Claude (Anthropic) for vision.
The **matcher** supports two providers, selectable via CLI:

| Provider | Flag | Default model | Notes |
|---|---|---|---|
| Anthropic | `--matcher-provider anthropic` | `claude-sonnet-4-6` | Default |
| OpenRouter | `--matcher-provider openrouter` | `google/gemini-2.0-flash-001` | Any model at [openrouter.ai/models](https://openrouter.ai/models) |

---

## Setup

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtualenv and install dependencies
uv sync

# Configure API keys
cp .env.example .env
# edit .env: add ANTHROPIC_API_KEY and optionally OPENROUTER_API_KEY
```

---

## Usage

### Full pipeline (stops for plan review)

```bash
uv run main.py \
  --video-url "https://youtube.com/watch?v=..." \
  --song-url  "https://youtube.com/watch?v=..." \
  --film "Licorice Pizza" \
  --song "Call it Fate, Call it Karma" \
  --song-start 15 \
  --song-end 105
```

### After reviewing / editing the plan

```bash
uv run main.py \
  --video-url "..." --song-url "..." \
  --film "Licorice Pizza" --song "Call it Fate, Call it Karma" \
  --render-only
```

### Use OpenRouter for matching

```bash
uv run main.py ... \
  --matcher-provider openrouter \
  --matcher-model "meta-llama/llama-3.3-70b-instruct"
```

---

## Configuration (`config.py`)

| Constant | Default | Description |
|---|---|---|
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model for vision + matching |
| `OPENROUTER_DEFAULT_MODEL` | `google/gemini-2.0-flash-001` | Default OpenRouter model |
| `SCENE_THRESHOLD` | `27.0` | PySceneDetect sensitivity (lower = more cuts) |
| `MIN_SCENE_DURATION` | `1.5s` | Discard scenes shorter than this |
| `N_FRAMES_PER_SCENE` | `3` | Frames sampled per scene for vision analysis |
| `SCENE_ANALYSIS_BATCH` | `5` | Scenes per Claude vision request |
| `WHISPER_MODEL` | `base` | faster-whisper model size |
| `MAX_REEL_DURATION` | `90s` | Hard cap on output video length |
| `SUBTITLE_COLOR` | `yellow` | Lyric subtitle colour |
| `SUBTITLE_FONTSIZE` | `60` | Subtitle font size (px) |

---

## Project structure

```
song-video-maker/
├── main.py                    # CLI entry point (argparse)
├── config.py                  # All constants and env vars
├── pyproject.toml             # uv/pip dependencies and project metadata
├── .env.example               # API key template
├── models/
│   ├── scene.py               # Scene (Pydantic)
│   └── lyrics.py              # LyricLine, MatchedSegment (Pydantic)
├── pipeline/
│   ├── downloader.py          # yt-dlp: video + audio download
│   ├── scene_detector.py      # PySceneDetect: cut detection
│   ├── frame_sampler.py       # OpenCV: frame extraction
│   ├── scene_analyzer.py      # Claude vision: scene description + film filter
│   ├── lyrics_extractor.py    # faster-whisper: lyrics + timestamps
│   ├── matcher.py             # LLM: scene-to-lyric plan
│   └── editor.py              # MoviePy + ffmpeg: final render
├── temp/                      # Downloaded files, frames (gitignored)
└── outputs/                   # Plans and rendered videos (gitignored)
```
