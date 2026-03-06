"""Detect scene cuts in a video using PySceneDetect."""
from pathlib import Path

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from config import SCENE_THRESHOLD, MIN_SCENE_DURATION
from models import Scene


def detect_scenes(
    video_path: Path,
    threshold: float = SCENE_THRESHOLD,
    min_duration: float = MIN_SCENE_DURATION,
) -> list[Scene]:
    """Detect scene cuts and return a list of Scene objects with timecodes."""
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video, show_progress=True)
    raw_scenes = scene_manager.get_scene_list()

    scenes: list[Scene] = []
    for i, (start, end) in enumerate(raw_scenes):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        duration = end_sec - start_sec

        if duration < min_duration:
            continue

        scenes.append(
            Scene(
                index=len(scenes),
                start_time=start_sec,
                end_time=end_sec,
            )
        )

    print(f"[scene_detector] Found {len(scenes)} scenes (≥{min_duration}s) from {len(raw_scenes)} total cuts")
    return scenes
