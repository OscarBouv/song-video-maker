# API Cost Estimates

Prices as of **March 2026**. All costs are estimates based on token-counting formulas from official documentation.

---

## Assumptions

| Parameter | Value | Note |
|---|---|---|
| Frame resolution | 1920×1080 | Typical YouTube compilation |
| Frames per scene | 3 | Default `--n-frames` |
| Scenes per vision batch | 5 | Default `SCENE_ANALYSIS_BATCH` |
| Lyrics source | LRCLIB (free) | Whisper fallback is local, also free |

---

## Model Pricing

| Model | Input | Output | Image tokens (1920×1080) |
|---|---|---|---|
| `claude-sonnet-4-6` | $3.00 / MTok | $15.00 / MTok | ~1,844 tok |
| `claude-opus-4-6` | $5.00 / MTok | $25.00 / MTok | ~1,844 tok |
| `google/gemini-2.0-flash-001` | $0.10 / MTok | $0.40 / MTok | ~1,548 tok |
| `meta-llama/llama-4-maverick` | $0.15 / MTok | $0.60 / MTok | ≤3,342 tok |

**Image token formulas:**
- **Claude** — resizes to fit 1568px long edge, then `(w × h) / 750` → 1920×1080 → 1568×882 → ~1,844 tok
- **Gemini** — 768×768 tiles × 258 tok/tile → ceil(1920/768)×ceil(1080/768) = 3×2 = 6 tiles → 1,548 tok
- **Llama 4 Maverick** — capped at 3,342 tok per image regardless of resolution

---

## Step 1 — Scene Analysis (Vision, per scene)

Each scene sends **3 frames + scene label text** to the vision model.

| | Tokens | Claude Sonnet 4.6 | Claude Opus 4.6 | Gemini Flash | Llama 4 Maverick |
|---|---|---|---|---|---|
| Image input (3 frames) | 5,532 / 4,644 / 10,026 | $0.01660 | $0.02766 | $0.000464 | $0.001504 |
| Text input (batch overhead ÷ 5) | ~60 tok | $0.00018 | $0.00030 | $0.000006 | $0.000009 |
| Output (JSON entry) | ~50 tok | $0.00075 | $0.00125 | $0.000020 | $0.000030 |
| **Total per scene** | | **$0.018** | **$0.029** | **$0.0005** | **$0.0015** |

### Scene analysis totals by pipeline size

| Scenes | Claude Sonnet 4.6 | Claude Opus 4.6 | Gemini Flash | Llama 4 Maverick |
|---|---|---|---|---|
| 50 | $0.88 | $1.46 | $0.025 | $0.077 |
| **100** | **$1.75** | **$2.92** | **$0.049** | **$0.154** |
| 150 | $2.63 | $4.38 | $0.074 | $0.231 |

> A 5–10 min YouTube compilation typically yields **80–130 scenes** after filtering.

---

## Step 2 — Scene-to-Lyric Matcher (Text, one call per pipeline run)

The matcher receives all scene descriptions + full lyrics, returns a JSON plan.

| Token estimate | ~6,000 input | ~2,000 output |
|---|---|---|
| Scene descriptions (100 scenes × ~20 tok) | 2,000 tok | — |
| System prompt + lyrics (~100 lines × 30 tok) | 4,000 tok | — |
| JSON plan output | — | 2,000 tok |

| Model | Input cost | Output cost | **Total** |
|---|---|---|---|
| `claude-sonnet-4-6` | $0.018 | $0.030 | **$0.048** |
| `claude-opus-4-6` | $0.030 | $0.050 | **$0.080** |
| `google/gemini-2.0-flash-001` | $0.0006 | $0.0008 | **$0.0014** |
| `meta-llama/llama-4-maverick` | $0.0009 | $0.0012 | **$0.0021** |

---

## Full Pipeline Cost (100 scenes)

| Analyzer | Matcher | Total | Notes |
|---|---|---|---|
| Claude Sonnet 4.6 | Claude Sonnet 4.6 | **~$1.80** | Default config |
| Claude Opus 4.6 | Claude Opus 4.6 | **~$3.00** | Highest quality |
| Gemini Flash | Gemini Flash | **~$0.05** | Cheapest option |
| Gemini Flash | Claude Sonnet 4.6 | **~$0.10** | Recommended hybrid |
| Llama 4 Maverick | Gemini Flash | **~$0.16** | Open model analysis |

> **Recommended hybrid**: use `--analyzer-provider openrouter --analyzer-model google/gemini-2.0-flash-001`
> for cheap scene analysis, keeping Claude Sonnet for the nuanced scene-to-lyric matching.

---

## Cache Savings

Scene analysis results are cached in `temp/cache/scene_analysis.json` (keyed by MD5 of frame bytes + film name).

| Run | Cost |
|---|---|
| First run (cold cache) | Full cost as above |
| Re-run, same video | **$0** for scene analysis |
| Re-run, different song / edit plan | **$0** for scene analysis + ~$0.05 for new matcher call |

Use the Batch API (`--provider anthropic` uses standard pricing; Anthropic Batch API offers 50% off but requires async polling — not currently implemented).

---

## Tips to Reduce Cost

- **Use `--n-frames 1`** — reduces vision tokens by 3×, at the cost of less accurate scene descriptions.
- **Use Gemini Flash for analysis** — ~35× cheaper than Claude Sonnet for vision, quality is comparable for simple scene description.
- **Pre-cut your video** — `--video-start` / `--video-end` reduces the number of scenes detected.
- **Cache is your friend** — the first run is expensive; all re-runs (plan edits, re-renders) cost almost nothing.
- **Batch API** — if you process many videos, Anthropic's Batch API halves the cost (not yet wired up).

---

*Prices sourced from [Anthropic Pricing](https://platform.claude.com/docs/en/about-claude/pricing), [Anthropic Vision Docs](https://platform.claude.com/docs/en/build-with-claude/vision), [OpenRouter](https://openrouter.ai). Always verify current rates before large-scale usage.*
