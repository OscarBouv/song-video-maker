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
WORKSPACES_DIR = ROOT_DIR / "workspaces"

CLIPS_DIR      = ROOT_DIR / "clips"   # video + scenes + frames + scene-analysis cache
SONGS_DIR      = ROOT_DIR / "songs"   # audio + lyrics + whisper cache

TEMP_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
WORKSPACES_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True)
SONGS_DIR.mkdir(exist_ok=True)

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
N_FRAMES_PER_SCENE = 1       # frames sampled per scene for vision analysis
SCENE_ANALYSIS_BATCH = 5     # scenes per vision request

# Whisper
WHISPER_MODEL = "base"       # base is fast and accurate enough for lyrics

# Instagram Publication
INSTAGRAM_USERNAME = os.getenv("INSTAGRAM_USERNAME")
INSTAGRAM_PASSWORD = os.getenv("INSTAGRAM_PASSWORD")
INSTAGRAM_SESSION_FILE = ROOT_DIR / "instagram_session.json"

# Output video
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30
MAX_REEL_DURATION = 9999     # effectively unlimited — natural limit is the song duration

# Subtitle style
SUBTITLE_COLOR        = "#FFD966"  # warm soft yellow — readable on dark backgrounds
SUBTITLE_BORDER_COLOR = "black"
SUBTITLE_BORDER_WIDTH = 3          # px — crisp outline instead of soft shadow only
SUBTITLE_FONTSIZE     = 56         # was 46 — larger for mobile readability
SUBTITLE_Y_RATIO      = 0.82       # vertical position (0 = top, 1 = bottom)


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


def _find_insert_font() -> str:
    """Find a clean, non-rounded font for the source credit insert (distinct from subtitle font)."""
    for path in [
        "/System/Library/Fonts/Supplemental/GillSans.ttc",
        "/Library/Fonts/GillSans.ttc",
        "/System/Library/Fonts/Supplemental/Futura.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if Path(path).exists():
            return path
    return _find_font("Arial")


def _discover_subtitle_fonts() -> dict[str, str]:
    """Return {display_name: font_path} for every available subtitle font on this machine.

    Uses explicit absolute paths so fallback logic in _find_font() doesn't pollute results
    with wrong fonts.  The order defines the display order in the UI.
    """
    candidates: list[tuple[str, list[str]]] = [
        ("Arial Rounded Bold", ["/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"]),
        ("Helvetica",          ["/System/Library/Fonts/Helvetica.ttc"]),
        ("Avenir Next",        ["/System/Library/Fonts/Avenir Next.ttc",
                                "/System/Library/Fonts/AvenirNext.ttc"]),
        ("Futura",             ["/System/Library/Fonts/Supplemental/Futura.ttc"]),
        ("Gill Sans",          ["/System/Library/Fonts/Supplemental/GillSans.ttc",
                                "/Library/Fonts/GillSans.ttc"]),
        ("Didot",              ["/System/Library/Fonts/Supplemental/Didot.ttc"]),
        ("Baskerville",        ["/System/Library/Fonts/Supplemental/Baskerville.ttc"]),
        ("Georgia",            ["/System/Library/Fonts/Supplemental/Georgia.ttf",
                                "/Library/Fonts/Georgia.ttf"]),
        ("Courier New",        ["/System/Library/Fonts/Supplemental/Courier New.ttf",
                                "/Library/Fonts/Courier New.ttf"]),
        ("Trebuchet MS",       ["/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
                                "/Library/Fonts/Trebuchet MS.ttf"]),
        # Google Fonts — present only if manually installed
        ("Sarabun",            ["/Library/Fonts/Sarabun-Bold.ttf",
                                "/Library/Fonts/Sarabun Bold.ttf",
                                "/Library/Fonts/Sarabun.ttf"]),
        ("Inter",              ["/Library/Fonts/Inter-Bold.ttf",
                                "/Library/Fonts/Inter Bold.ttf",
                                "/Library/Fonts/Inter.ttf"]),
    ]
    result: dict[str, str] = {}
    for display_name, paths in candidates:
        for path in paths:
            if Path(path).exists():
                result[display_name] = path
                break
    return result


# Reel fade-in / fade-out (video + audio)
VIDEO_FADE_DURATION = 0.5   # seconds — video fade to/from black at both ends
AUDIO_FADE_DURATION = 1.0   # seconds — audio is faded slightly longer for a smoother feel

# Insert overlay — top-left corner, fades in at INSERT_START_T and disappears at INSERT_END_T
INSERT_FONT         = _find_insert_font()
INSERT_FONTSIZE     = 27
INSERT_COLOR        = "#FFD966@0.72" # same warm yellow as subtitles; multiplied by alpha expression
INSERT_BORDER_WIDTH = 1
INSERT_X_RATIO      = 0.048          # left margin as fraction of output width
INSERT_Y_TOP_RATIO  = 0.033          # top margin as fraction of output height
INSERT_START_T      = 2.0            # seconds — fade-in begins
INSERT_FADE_T       = 0.5            # seconds — fade-in duration
INSERT_END_T        = 7.0            # seconds — fully gone (5s total window: 2→7)
INSERT_FADE_OUT_T   = 0.5            # seconds — fade-out duration before INSERT_END_T

# Subtitle font options (populated at startup from available system fonts)
SUBTITLE_FONTS: dict[str, str] = _discover_subtitle_fonts()
