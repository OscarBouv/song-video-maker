from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).parent
TEMP_DIR = ROOT_DIR / "temp"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FRAMES_DIR = TEMP_DIR / "frames"

TEMP_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)

# OpenRouter (all AI calls go through OpenRouter)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(str, Enum):
    """Curated OpenRouter models for scene analysis and scene-to-lyric matching."""
    claude_sonnet  = "anthropic/claude-sonnet-4-6"           # best reasoning, highest quality
    claude_haiku   = "anthropic/claude-haiku-4-5-20251001"   # fast, cheap, still capable
    gemini_flash   = "google/gemini-2.0-flash-001"           # very cheap vision, good quality
    gemini_pro     = "google/gemini-2.5-pro-preview"         # high quality, slower
    llama_maverick = "meta-llama/llama-4-maverick"           # open model, vision-capable


# Default models per pipeline step
DEFAULT_ANALYZER_MODEL = OpenRouterModel.gemini_flash   # cheap + fast for batch vision analysis
DEFAULT_MATCHER_MODEL  = OpenRouterModel.claude_sonnet  # best reasoning for scene-to-lyric plan

# Scene detection
SCENE_THRESHOLD = 27.0       # PySceneDetect ContentDetector threshold
MIN_SCENE_DURATION = 1.5     # seconds — discard very short scenes
N_FRAMES_PER_SCENE = 3       # frames sampled per scene for vision analysis
SCENE_ANALYSIS_BATCH = 5     # scenes per vision request

# Whisper
WHISPER_MODEL = "base"       # base is fast and accurate enough for lyrics

# Output video
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30
MAX_REEL_DURATION = 90       # seconds

# Subtitle style
SUBTITLE_COLOR = "yellow"
SUBTITLE_FONTSIZE = 46              # smaller = more subtle
SUBTITLE_Y_RATIO = 0.82            # vertical position (0 = top, 1 = bottom)


def _find_ffmpeg() -> tuple[str, str]:
    """Return (ffmpeg_bin, ffprobe_bin) preferring ffmpeg-full (has drawtext/libass)."""
    full_prefix = Path("/opt/homebrew/opt/ffmpeg-full/bin")
    if (full_prefix / "ffmpeg").exists():
        return str(full_prefix / "ffmpeg"), str(full_prefix / "ffprobe")
    # Fallback: whatever is on PATH
    return "ffmpeg", "ffprobe"


FFMPEG_BIN, FFPROBE_BIN = _find_ffmpeg()


def _find_font(name: str) -> str:
    """Resolve a font name to an absolute .ttf/.otf path.

    PIL's ImageFont.truetype() on macOS cannot look up fonts by name — it needs
    a real file path. This searches the standard macOS (and Linux) font directories
    and falls back to system fonts when the requested font isn't found.
    """
    # Build candidate paths from the bare name and common naming variants
    variants = [name, name.replace("-", " "), name.replace("-", "")]
    search_dirs = [
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path("/System/Library/Fonts"),
        Path("/usr/share/fonts/truetype"),          # Linux
        Path("/usr/share/fonts/truetype/dejavu"),   # Linux/Debian
    ]
    for directory in search_dirs:
        if not directory.exists():
            continue
        for variant in variants:
            for ext in (".ttf", ".otf", ".ttc"):
                candidate = directory / f"{variant}{ext}"
                if candidate.exists():
                    return str(candidate)

    # Generic bold fallbacks guaranteed to exist on macOS / most Linux distros
    fallbacks = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in fallbacks:
        if Path(path).exists():
            return path

    # Last resort — return the bare name and let PIL raise a clear error
    return name


SUBTITLE_FONT = _find_font("Arial Rounded Bold")  # softer, more modern than standard Arial Bold
