"""Extract representative frames from each scene using OpenCV."""
from pathlib import Path

import cv2

from config import FRAMES_DIR, N_FRAMES_PER_SCENE
from models import Scene


def sample_frames(
    video_path: Path,
    scene: Scene,
    n_frames: int = N_FRAMES_PER_SCENE,
    output_dir: Path = FRAMES_DIR,
) -> list[str]:
    """Extract n_frames evenly spaced frames from a scene. Returns list of file paths."""
    scene_dir = output_dir / f"scene_{scene.index:04d}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # Compute evenly spaced timestamps within the scene
    duration = scene.end_time - scene.start_time
    if n_frames == 1:
        timestamps = [scene.start_time + duration / 2]
    else:
        step = duration / (n_frames + 1)
        timestamps = [scene.start_time + step * (i + 1) for i in range(n_frames)]

    frame_paths: list[str] = []
    for j, ts in enumerate(timestamps):
        frame_number = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = scene_dir / f"frame_{j:02d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_paths.append(str(out_path))

    cap.release()
    return frame_paths


def sample_all_scenes(
    video_path: Path,
    scenes: list[Scene],
    n_frames: int = N_FRAMES_PER_SCENE,
) -> list[Scene]:
    """Sample frames for all scenes in-place and return updated scenes."""
    for scene in scenes:
        scene.frames = sample_frames(video_path, scene, n_frames)
    print(f"[frame_sampler] Sampled {n_frames} frames for {len(scenes)} scenes")
    return scenes
