# song-video-maker

Generates aesthetic 9:16 Instagram Reels from movie clip compilations + songs. Matches scenes to lyrics via vision/text LLMs, renders with word-synced subtitles.

## Setup

```bash
uv sync
cp .env.example .env   # add OPENROUTER_API_KEY
```

Run the web UI:
```bash
uv run uvicorn web_app:app --reload --port 8000
```

Run the CLI:
```bash
uv run song-video-maker --help
```

## Project structure

```
pipeline/          # 8 pipeline steps (one file each)
models/            # Pydantic models: Scene, LyricLine, MatchedSegment
workspaces/        # One dir per film+song pair (gitignored)
web_app.py         # FastAPI backend + SSE streaming
static/index.html  # Alpine.js + Tailwind + SortableJS SPA
config.py          # All constants, env vars, paths, models enum
main.py            # Thin entry point → pipeline/cli.py
```

## Pipeline steps (in order)

| Step | File | What it does |
|------|------|-------------|
| download-video | `downloader.py` | yt-dlp, optionally trimmed |
| download-audio | `downloader.py` | yt-dlp audio |
| detect-scenes | `scene_detector.py` | PySceneDetect hard-cut detection |
| sample-frames | `frame_sampler.py` | OpenCV frame extraction (3/scene) |
| analyze-scenes | `scene_analyzer.py` | Vision LLM via OpenRouter, batched 5 |
| extract-lyrics | `lyrics_extractor.py` | LRCLIB API → faster-whisper refinement |
| generate-plan | `matcher.py` | LLM scene-to-lyric mapping |
| render | `editor.py` | Single ffmpeg pass, drawtext subtitles |

Recommended flow: run all steps up to `generate-plan`, review `{slug}_plan_readable.txt`, optionally use `edit-plan`, then `render`.

## Workspace layout

All state lives under `workspaces/{film}_{song}/`:
- `state/` — scenes.json, lyrics.json, audio_path.txt, video_path.txt
- `frames/` — JPEG frames per scene
- `cache/` — scene_analysis_{model}.json, llm_responses_{model}.json
- `{slug}_plan.json` + `{slug}_plan_readable.txt`
- `{slug}_reel.mp4`

## Key CLI flags

```bash
# Full pipeline (pauses after plan)
uv run song-video-maker run \
  --video-url <url> --song-url <url> \
  --film "Licorice Pizza" --song "Call It Fate Call It Karma" \
  --characters "Alana,Gary" --matcher-model claude_sonnet

# Re-render from existing plan
uv run song-video-maker run ... --render-only

# Edit plan with natural language
uv run song-video-maker edit-plan \
  --film "..." --song "..." \
  --instruction "swap clip 3 with clip 7"

# Status of all workspaces
uv run song-video-maker status
```

## LLM models (config.py `OpenRouterModel`)

| Alias | Model ID | Use |
|-------|----------|-----|
| `claude_sonnet` | anthropic/claude-sonnet-4-6 | matching (best reasoning) |
| `claude_haiku` | anthropic/claude-haiku-4-5-20251001 | fast/cheap |
| `gemini_flash` | google/gemini-2.0-flash-001 | scene analysis default |
| `gemini_pro` | google/gemini-2.5-pro-preview | high quality |
| `llama_maverick` | meta-llama/llama-4-maverick | open model |

All calls go through OpenRouter using the OpenAI SDK pointed at `https://openrouter.ai/api/v1`. Responses are cached to disk — re-runs are free.

## Output spec

- Resolution: 1080×1920 (9:16 portrait)
- Max duration: 90s
- Subtitles: white Arial Rounded Bold, size 56, black stroke, y=82% height
- ffmpeg preferred path: `/opt/homebrew/opt/ffmpeg-full/bin` (needs drawtext filter)

## Data models (`models/`)

- `Scene` — index, start/end times, description, emotion, shot_type, lighting, visual_power (1-5), is_film_related, is_usable
- `LyricLine` — start/end, text, word-level timestamps
- `MatchedSegment` — links a scene slice to a lyric window, with trim_start/trim_end offsets

## Web API routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/workspaces` | list all |
| GET | `/api/workspace/{slug}` | status |
| GET | `/api/workspace/{slug}/scenes` | scene list + thumbnails |
| GET | `/api/workspace/{slug}/plan` | plan + metadata |
| PUT | `/api/workspace/{slug}/plan` | save reordered plan |
| POST | `/api/workspace/create` (SSE) | run full pipeline |
| POST | `/api/pipeline/run` (SSE) | run one step |
| POST | `/api/workspace/{slug}/edit-plan` (SSE) | LLM plan edit |
| POST | `/api/workspace/{slug}/render` (SSE) | render reel |
| GET | `/api/workspace/{slug}/reel` | stream final MP4 |
| GET | `/api/frame?path=...` | serve frame JPEGs |

## Common gotchas

- `OPENROUTER_API_KEY` must be set; the code also checks `ANTHROPIC_API_KEY` in comments but all live LLM calls use OpenRouter.
- ffmpeg must support `drawtext` filter — the Homebrew `ffmpeg-full` formula includes it; the default `ffmpeg` formula may not.
- Scene cache is keyed by model name — switching `--analyzer-model` re-runs analysis (cached separately per model).
- `edit-plan` always auto-backs up the plan before modifying it.
- `lyrics_extractor.py` tries LRCLIB first; faster-whisper is only used if LRCLIB returns no results.