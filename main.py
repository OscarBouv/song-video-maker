"""song-video-maker CLI — generate aesthetic Instagram Reels from movie clips + songs."""
import argparse
import json
import re
import sys
from pathlib import Path

from config import TEMP_DIR, OUTPUTS_DIR, N_FRAMES_PER_SCENE, CLAUDE_MODEL, OPENROUTER_DEFAULT_MODEL


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def run_pipeline(args: argparse.Namespace) -> None:
    slug = slugify(f"{args.film}_{args.song}")
    plan_path = OUTPUTS_DIR / f"{slug}_plan.json"
    output_path = OUTPUTS_DIR / f"{slug}_reel.mp4"
    video_dl_path = TEMP_DIR / "video.mp4"
    audio_dl_path = TEMP_DIR / "audio.mp3"

    # ── Render-only mode ──────────────────────────────────────────────────────
    if args.render_only:
        if not plan_path.exists():
            print(f"[main] Error: plan file not found at {plan_path}")
            print("       Run without --render-only first to generate the plan.")
            sys.exit(1)
        if not video_dl_path.exists() or not audio_dl_path.exists():
            print("[main] Error: temp/video.mp4 or temp/audio.mp3 not found.")
            print("       Re-run without --render-only to download files again.")
            sys.exit(1)

        from pipeline.scene_detector import detect_scenes
        from pipeline.matcher import load_plan
        from pipeline.editor import render_video
        from models import LyricLine

        print("[main] Loading existing plan and rendering...")
        scenes = detect_scenes(video_dl_path)  # need scenes for metadata
        with open(plan_path) as f:
            plan_data = json.load(f)
        # Reconstruct LyricLine objects from plan (lyric_lines embedded in segments)
        segments = []
        from models import MatchedSegment
        for entry in plan_data:
            segments.append(MatchedSegment.from_dict(entry))

        render_video(segments, scenes, video_dl_path, audio_dl_path, output_path)
        return

    # ── Full pipeline ─────────────────────────────────────────────────────────
    from pipeline.downloader import download_video, download_audio
    from pipeline.scene_detector import detect_scenes
    from pipeline.frame_sampler import sample_all_scenes
    from pipeline.scene_analyzer import analyze_scenes
    from pipeline.lyrics_extractor import extract_lyrics
    from pipeline.matcher import generate_plan

    # Step 1: Download
    print("\n── Step 1/6: Downloading video ──────────────────────────────")
    video_path = download_video(args.video_url, TEMP_DIR)

    print("\n── Step 2/6: Downloading audio ──────────────────────────────")
    audio_path = download_audio(
        args.song_url,
        TEMP_DIR,
        start_sec=args.song_start,
        end_sec=args.song_end,
    )

    # Step 2: Scene detection
    print("\n── Step 3/6: Detecting scenes ───────────────────────────────")
    scenes = detect_scenes(video_path)
    print(f"  → {len(scenes)} scenes detected")

    # Step 3: Frame sampling + AI analysis
    print("\n── Step 4/6: Sampling frames + AI scene analysis ────────────")
    scenes = sample_all_scenes(video_path, scenes, n_frames=args.n_frames)
    scenes = analyze_scenes(scenes, film_name=args.film)

    # Step 4: Lyrics extraction
    print("\n── Step 5/6: Extracting lyrics ──────────────────────────────")
    lyrics = extract_lyrics(audio_path, start_sec=args.song_start, end_sec=args.song_end)
    if not lyrics:
        print("[main] Error: no lyrics extracted. Check your audio file.")
        sys.exit(1)

    # Step 5: AI matching + plan generation
    print("\n── Step 6/6: Generating scene-to-lyric plan ─────────────────")
    generate_plan(
        scenes=scenes,
        lyrics=lyrics,
        film_name=args.film,
        song_name=args.song,
        output_dir=OUTPUTS_DIR,
        slug=slug,
        provider=args.matcher_provider,
        model=args.matcher_model,
    )

    print("\n✓ Pipeline complete. Next steps:")
    print(f"  1. Review  : outputs/{slug}_plan_readable.txt")
    print(f"  2. Edit    : outputs/{slug}_plan.json  (if needed)")
    print(f"  3. Render  : python main.py --film '{args.film}' --song '{args.song}'")
    print(f"               --video-url '{args.video_url}' --song-url '{args.song_url}'")
    print(f"               --render-only")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an aesthetic Instagram Reel from movie clips + a song."
    )
    parser.add_argument("--video-url", required=True, help="YouTube URL of the movie clip compilation")
    parser.add_argument("--song-url", required=True, help="YouTube URL of the song")
    parser.add_argument("--film", required=True, help="Film name (e.g. 'Licorice Pizza')")
    parser.add_argument("--song", required=True, help="Song name (e.g. 'Call it Fate, Call it Karma')")
    parser.add_argument("--song-start", type=float, default=None, help="Start time in song (seconds)")
    parser.add_argument("--song-end", type=float, default=None, help="End time in song (seconds)")
    parser.add_argument("--n-frames", type=int, default=N_FRAMES_PER_SCENE, help="Frames sampled per scene for AI analysis")
    parser.add_argument(
        "--matcher-provider",
        choices=["anthropic", "openrouter"],
        default="anthropic",
        help="LLM provider for scene-to-lyric matching (default: anthropic)",
    )
    parser.add_argument(
        "--matcher-model",
        default=None,
        help=(
            f"Model ID override for matcher. "
            f"Anthropic default: {CLAUDE_MODEL}. "
            f"OpenRouter default: {OPENROUTER_DEFAULT_MODEL}. "
            "See https://openrouter.ai/models for OpenRouter IDs."
        ),
    )
    parser.add_argument("--render-only", action="store_true", help="Skip to render using existing plan.json")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
