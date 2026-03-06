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

# Claude (used for scene analysis + matching by default)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-6"

# OpenRouter (alternative provider for matcher)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "google/gemini-2.0-flash-001"  # fast + cheap default

# Scene detection
SCENE_THRESHOLD = 27.0       # PySceneDetect ContentDetector threshold
MIN_SCENE_DURATION = 1.5     # seconds — discard very short scenes
N_FRAMES_PER_SCENE = 3       # frames sampled per scene for vision analysis
SCENE_ANALYSIS_BATCH = 5     # scenes per Claude vision request

# Whisper
WHISPER_MODEL = "base"       # base is fast and accurate enough for lyrics

# Output video
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30
MAX_REEL_DURATION = 90       # seconds

# Subtitle style
SUBTITLE_COLOR = "yellow"
SUBTITLE_FONTSIZE = 60
SUBTITLE_FONT = "Arial-Bold"
SUBTITLE_POSITION = ("center", 0.80)  # 80% down the frame
