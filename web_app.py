"""FastAPI web app for song-video-maker — pipeline UI + plan editor."""
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from config import WORKSPACES_DIR
from models import Scene, MatchedSegment

app = FastAPI(title="song-video-maker")

STATIC_DIR = ROOT_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ws_status(ws_dir: Path) -> dict:
    state = ws_dir / "state"
    cfg: dict = {}
    cfg_file = ws_dir / "workspace.json"
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text())
        except Exception:
            pass
    steps = []
    if (state / "video_path.txt").exists():        steps.append("video")
    if (state / "audio_path.txt").exists():        steps.append("audio")
    if (state / "scenes.json").exists():           steps.append("scenes")
    if (state / "lyrics.json").exists():           steps.append("lyrics")
    if next(ws_dir.glob("*_plan.json"), None):     steps.append("plan")
    if next(ws_dir.glob("*_reel.mp4"), None):      steps.append("reel")
    return {
        "slug": ws_dir.name,
        "film": cfg.get("film", ws_dir.name),
        "song": cfg.get("song", ""),
        "artist": cfg.get("artist", ""),
        "created_at": cfg.get("created_at", ""),
        "config": cfg,
        "steps_complete": steps,
        "has_reel": "reel" in steps,
    }


async def _stream_cli(args: list[str]):
    """Run a CLI subcommand and yield SSE-formatted log lines."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = await asyncio.create_subprocess_exec(
        "uv", "run", "song-video-maker", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(ROOT_DIR),
        env=env,
    )
    assert proc.stdout is not None
    try:
        async for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                yield f"data: {json.dumps({'log': text})}\n\n"
    finally:
        if proc.returncode is None:
            proc.terminate()
        code = await proc.wait()
        yield f"data: {json.dumps({'done': True, 'exit_code': code})}\n\n"


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/api/workspaces")
def list_workspaces():
    if not WORKSPACES_DIR.exists():
        return []
    dirs = sorted(d for d in WORKSPACES_DIR.iterdir() if d.is_dir())
    return [_ws_status(d) for d in dirs]


@app.get("/api/workspace/{slug}")
def get_workspace(slug: str):
    ws_dir = WORKSPACES_DIR / slug
    if not ws_dir.exists():
        raise HTTPException(404, f"Workspace '{slug}' not found")
    result = _ws_status(ws_dir)
    scenes_file = ws_dir / "state" / "scenes.json"
    if scenes_file.exists():
        try:
            scenes = [Scene.model_validate(d) for d in json.loads(scenes_file.read_text())]
            result["scenes_count"] = len(scenes)
            result["scenes_usable"] = sum(1 for s in scenes if s.is_usable)
        except Exception:
            result["scenes_count"] = 0
            result["scenes_usable"] = 0
    lyrics_file = ws_dir / "state" / "lyrics.json"
    if lyrics_file.exists():
        try:
            result["lyrics_count"] = len(json.loads(lyrics_file.read_text()))
        except Exception:
            result["lyrics_count"] = 0
    return result


@app.get("/api/workspace/{slug}/scenes")
def get_scenes(slug: str):
    ws_dir = WORKSPACES_DIR / slug
    scenes_file = ws_dir / "state" / "scenes.json"
    if not scenes_file.exists():
        raise HTTPException(404, "No scenes.json — run detect-scenes first")
    scenes = [Scene.model_validate(d) for d in json.loads(scenes_file.read_text())]
    return [
        {
            "index": s.index,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "duration": s.duration,
            "description": s.description,
            "characters_present": s.characters_present,
            "emotion": s.emotion,
            "shot_type": s.shot_type,
            "lighting": s.lighting,
            "visual_power": s.visual_power,
            "is_film_related": s.is_film_related,
            "is_aesthetic": s.is_aesthetic,
            "is_usable": s.is_usable,
            "thumbnail": f"/api/frame?path={s.frames[0]}" if s.frames else None,
        }
        for s in scenes
    ]


@app.get("/api/workspace/{slug}/plan")
def get_plan(slug: str):
    ws_dir = WORKSPACES_DIR / slug
    plan_file = next(ws_dir.glob("*_plan.json"), None)
    if not plan_file:
        raise HTTPException(404, "No plan — run generate-plan first")
    segments_raw = json.loads(plan_file.read_text())
    scenes_file = ws_dir / "state" / "scenes.json"
    scene_by_index: dict[int, Scene] = {}
    if scenes_file.exists():
        for d in json.loads(scenes_file.read_text()):
            s = Scene.model_validate(d)
            scene_by_index[s.index] = s
    result = []
    for i, seg_dict in enumerate(segments_raw):
        seg = MatchedSegment.from_dict(seg_dict)
        scene = scene_by_index.get(seg.scene_index)
        thumbnail = f"/api/frame?path={scene.frames[0]}" if scene and scene.frames else None
        result.append({
            "clip_number": i + 1,
            "scene_index": seg.scene_index,
            "scene_trim_start": seg.scene_trim_start,
            "scene_trim_end": seg.scene_trim_end,
            "song_start": seg.song_start,
            "song_end": seg.song_end,
            "lyric_lines": [ll.model_dump() for ll in seg.lyric_lines],
            "scene": {
                "description": scene.description,
                "emotion": scene.emotion,
                "visual_power": scene.visual_power,
                "characters_present": scene.characters_present,
                "duration": scene.duration,
                "is_usable": scene.is_usable,
            } if scene else None,
            "thumbnail": thumbnail,
            "_raw": seg_dict,
        })
    return result


@app.put("/api/workspace/{slug}/plan")
async def save_plan(slug: str, body: dict):
    ws_dir = WORKSPACES_DIR / slug
    plan_file = next(ws_dir.glob("*_plan.json"), None)
    if not plan_file:
        raise HTTPException(404, "No plan file found")
    segments = body.get("segments", [])
    if not segments:
        raise HTTPException(400, "No segments provided")
    plan_file.write_text(json.dumps(segments, indent=2))
    return {"ok": True}


@app.get("/api/workspace/{slug}/reel")
def serve_reel(slug: str):
    ws_dir = WORKSPACES_DIR / slug
    reel = next(ws_dir.glob("*_reel.mp4"), None)
    if not reel:
        raise HTTPException(404, "No reel found — render first")
    return FileResponse(str(reel), media_type="video/mp4", headers={
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
    })


@app.get("/api/frame")
def serve_frame(path: str):
    p = Path(path)
    if not p.exists() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(404, "Frame not found")
    return FileResponse(str(p), media_type="image/jpeg")


@app.post("/api/workspace/create")
async def create_workspace(body: dict):
    """Run the full pipeline (up to plan) for a new workspace, streaming logs."""
    film = body.get("film", "")
    song = body.get("song", "")
    video_url = body.get("video_url", "")
    song_url = body.get("song_url", "")
    if not (film and song and video_url and song_url):
        raise HTTPException(400, "film, song, video_url, song_url are required")
    args = ["run", "--film", film, "--song", song,
            "--video-url", video_url, "--song-url", song_url]
    for field, flag in [
        ("artist", "--artist"), ("characters", "--characters"),
        ("video_start", "--video-start"), ("video_end", "--video-end"),
        ("song_start", "--song-start"), ("song_end", "--song-end"),
        ("analyzer_model", "--analyzer-model"), ("matcher_model", "--matcher-model"),
    ]:
        val = body.get(field)
        if val is not None and str(val).strip():
            args.extend([flag, str(val)])
    return StreamingResponse(_stream_cli(args), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/api/pipeline/run")
async def run_pipeline_step(body: dict):
    """Run a single pipeline step, streaming logs as SSE."""
    film = body.get("film", "")
    song = body.get("song", "")
    step = body.get("step", "")
    params: dict = body.get("params", {})
    if not (film and song and step):
        raise HTTPException(400, "film, song, and step are required")
    args = [step, "--film", film, "--song", song]
    for k, v in params.items():
        if v is None or str(v).strip() == "":
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                args.append(flag)
        else:
            args.extend([flag, str(v)])
    return StreamingResponse(_stream_cli(args), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/api/workspace/{slug}/edit-plan")
async def edit_plan_stream(slug: str, body: dict):
    """Run edit-plan and stream output as SSE."""
    ws_dir = WORKSPACES_DIR / slug
    cfg_file = ws_dir / "workspace.json"
    if not cfg_file.exists():
        raise HTTPException(404, f"Workspace '{slug}' not found")
    cfg = json.loads(cfg_file.read_text())
    instruction = body.get("instruction", "").strip()
    if not instruction:
        raise HTTPException(400, "instruction is required")
    args = ["edit-plan", "--film", cfg["film"], "--song", cfg["song"],
            "--instruction", instruction]
    if body.get("model"):
        args.extend(["--model", body["model"]])
    if body.get("auto_render"):
        args.append("--auto-render")
    return StreamingResponse(_stream_cli(args), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/api/workspace/{slug}/plan/history")
def get_plan_history(slug: str):
    """List all archived plans for this workspace, newest first."""
    import re
    ws_dir = WORKSPACES_DIR / slug
    history_dir = ws_dir / "plan_history"
    if not history_dir.exists():
        return []

    entries = []
    for p in sorted(history_dir.glob("*_plan_*.json"), reverse=True):
        # Skip readable text companions
        if "_readable" in p.name:
            continue
        # Extract timestamp from filename: {slug}_plan_{YYYYMMDD_HHMMSS}.json
        m = re.search(r"(\d{8}_\d{6})", p.stem)  # matches YYYYMMDD_HHMMSS (ignores optional _ms suffix)
        if not m:
            continue
        stamp = m.group(1)
        try:
            from datetime import datetime
            dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
            display = dt.strftime("%b %d, %Y · %H:%M")
            iso = dt.isoformat()
        except Exception:
            display = stamp
            iso = stamp

        clip_count = 0
        try:
            clip_count = len(json.loads(p.read_text()))
        except Exception:
            pass

        readable_name = p.stem + "_readable.txt"
        entries.append({
            "filename": p.name,
            "timestamp": iso,
            "display": display,
            "clip_count": clip_count,
            "has_readable": (history_dir / readable_name).exists(),
        })

    return entries


@app.post("/api/workspace/{slug}/plan/restore")
async def restore_plan(slug: str, body: dict):
    """Restore a plan from history, archiving the current plan first."""
    from pipeline.matcher import archive_existing_plan
    ws_dir = WORKSPACES_DIR / slug
    filename = body.get("filename", "").strip()
    if not filename:
        raise HTTPException(400, "filename is required")

    history_file = ws_dir / "plan_history" / filename
    if not history_file.exists():
        raise HTTPException(404, f"History file '{filename}' not found")

    plan_file = next(ws_dir.glob("*_plan.json"), None)
    if not plan_file:
        # Infer plan filename from slug
        cfg_file = ws_dir / "workspace.json"
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text())
            import re
            slug_str = re.sub(r"[^a-z0-9]+", "_", f"{cfg.get('film','')}_{cfg.get('song','')}".lower()).strip("_")
            plan_file = ws_dir / f"{slug_str}_plan.json"
        else:
            raise HTTPException(404, "No plan file found to replace")

    # Archive current plan first (so you can always undo the restore)
    archive_existing_plan(ws_dir, plan_file.stem.replace("_plan", ""))
    plan_file.write_bytes(history_file.read_bytes())

    # Restore matching readable if it exists
    readable_name = history_file.stem + "_readable.txt"
    readable_src = ws_dir / "plan_history" / readable_name
    if readable_src.exists():
        readable_dst = ws_dir / (plan_file.stem + "_readable.txt")
        readable_dst.write_bytes(readable_src.read_bytes())

    return {"ok": True, "restored": filename}


@app.get("/api/workspace/{slug}/plan/audit")
def audit_plan(slug: str):
    """Return a full timing audit: per-clip video positions, gaps, and duration checks."""
    ws_dir = WORKSPACES_DIR / slug
    plan_file = next(ws_dir.glob("*_plan.json"), None)
    if not plan_file:
        raise HTTPException(404, "No plan found — run generate-plan first")

    segments_raw = json.loads(plan_file.read_text())

    scenes_file = ws_dir / "state" / "scenes.json"
    scene_map: dict[int, Scene] = {}
    if scenes_file.exists():
        for d in json.loads(scenes_file.read_text()):
            s = Scene.model_validate(d)
            scene_map[s.index] = s

    # Probe audio duration
    audio_duration = 0.0
    audio_path_file = ws_dir / "state" / "audio_path.txt"
    if audio_path_file.exists():
        try:
            from pipeline.editor import _probe_duration
            ap = Path(audio_path_file.read_text().strip())
            if ap.exists():
                audio_duration = _probe_duration(ap)
        except Exception:
            pass

    # ── Per-clip analysis ──────────────────────────────────────────────────────
    running_time = 0.0
    clips_detail = []
    issues: list[dict] = []

    for i, seg_dict in enumerate(segments_raw):
        seg = MatchedSegment.from_dict(seg_dict)
        scene = scene_map.get(seg.scene_index)

        trim_dur  = (seg.scene_trim_end - seg.scene_trim_start) if seg.scene_trim_end >= 0 else 0.0
        plan_dur  = (seg.song_end - seg.song_start)             if seg.song_start >= 0    else trim_dur
        clip_dur  = trim_dur if trim_dur > 0 else plan_dur

        video_out_start = round(running_time, 4)
        video_out_end   = round(running_time + clip_dur, 4)

        # Drift: how far the video position has drifted from the audio plan position
        drift = round(running_time - seg.song_start, 4) if seg.song_start >= 0 else 0.0

        # Duration mismatch between plan and trim
        dur_ok = abs(plan_dur - trim_dur) < 0.05 if (plan_dur > 0 and trim_dur > 0) else None

        # Source overflow
        src_ok = True
        if scene and seg.scene_trim_end >= 0:
            scene_dur = scene.end_time - scene.start_time
            if seg.scene_trim_end > scene_dur + 0.05:
                src_ok = False
                issues.append({
                    "clip": i + 1, "severity": "error", "type": "source_overflow",
                    "msg": f"Clip {i+1} trim_end {seg.scene_trim_end:.2f}s > scene {seg.scene_index} duration {scene_dur:.2f}s",
                })

        if dur_ok is False:
            issues.append({
                "clip": i + 1, "severity": "warning", "type": "duration_mismatch",
                "msg": f"Clip {i+1}: plan={plan_dur:.3f}s trim={trim_dur:.3f}s (Δ{abs(plan_dur-trim_dur):.3f}s)",
            })

        if abs(drift) > 0.1:
            issues.append({
                "clip": i + 1, "severity": "warning", "type": "drift",
                "msg": f"Clip {i+1}: audio pos {seg.song_start:.2f}s ≠ video pos {running_time:.2f}s (drift {drift:+.2f}s)",
            })

        clips_detail.append({
            "n":               i + 1,
            "scene_index":     seg.scene_index,
            "song_start":      round(seg.song_start, 3),
            "song_end":        round(seg.song_end, 3),
            "plan_dur":        round(plan_dur, 3),
            "trim_dur":        round(trim_dur, 3),
            "dur_ok":          dur_ok,
            "video_out_start": video_out_start,
            "video_out_end":   video_out_end,
            "drift":           drift,
            "src_ok":          src_ok,
            "lyric":           " · ".join(l.get("text","") for l in seg_dict.get("lyric_lines",[]) if l.get("text")),
        })
        running_time += clip_dur

    plan_total = round(running_time, 4)
    end_gap    = round(audio_duration - plan_total, 4) if audio_duration > 0 else 0.0

    if audio_duration > 0 and abs(end_gap) > 0.1:
        issues.append({
            "clip": None, "severity": "error", "type": "total_gap",
            "msg": f"Plan total {plan_total:.3f}s ≠ audio {audio_duration:.3f}s  (gap {end_gap:+.3f}s)",
        })

    # ── Timeline gaps (holes in the song_start/song_end sequence) ─────────────
    timeline_gaps: list[dict] = []
    sorted_segs = [MatchedSegment.from_dict(d) for d in segments_raw if d.get("song_start", -1) >= 0]
    sorted_segs.sort(key=lambda s: s.song_start)
    cursor = 0.0
    for seg in sorted_segs:
        if seg.song_start - cursor > 0.1:
            timeline_gaps.append({"from": round(cursor,3), "to": round(seg.song_start,3),
                                   "dur": round(seg.song_start - cursor, 3)})
            issues.append({
                "clip": None, "severity": "error", "type": "timeline_gap",
                "msg": f"Timeline gap {cursor:.2f}s → {seg.song_start:.2f}s  ({seg.song_start - cursor:.2f}s uncovered)",
            })
        cursor = max(cursor, seg.song_end)
    if audio_duration > 0 and audio_duration - cursor > 0.1:
        timeline_gaps.append({"from": round(cursor,3), "to": round(audio_duration,3),
                               "dur": round(audio_duration - cursor, 3)})

    errors   = sum(1 for x in issues if x["severity"] == "error")
    warnings = sum(1 for x in issues if x["severity"] == "warning")

    return {
        "audio_duration": round(audio_duration, 3),
        "plan_total":     plan_total,
        "end_gap":        end_gap,
        "clips":          len(clips_detail),
        "timeline_gaps":  timeline_gaps,
        "issues":         issues,
        "errors":         errors,
        "warnings":       warnings,
        "ok":             errors == 0 and abs(end_gap) < 0.1,
        "clips_detail":   clips_detail,
    }


@app.get("/api/workspace/{slug}/video")
def serve_video(slug: str):
    """Stream the source video with byte-range support for in-browser clip preview."""
    ws_dir = WORKSPACES_DIR / slug
    video_path_file = ws_dir / "state" / "video_path.txt"
    if not video_path_file.exists():
        raise HTTPException(404, "No video — run download-video first")
    video_path = Path(video_path_file.read_text().strip())
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {video_path}")
    return FileResponse(
        str(video_path), media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache"},
    )


@app.post("/api/workspace/{slug}/plan/commit")
async def commit_plan(slug: str, body: dict):
    """Rebuild plan.json from clip-editor state.

    Accepts an ordered array of clips (scene_index, scene_trim_start, scene_trim_end,
    song_start, song_end).  For each clip, lyric_lines are re-synced from the current
    lyrics.json based on the song_start/song_end window.  The current plan is archived
    to plan_history/ before being overwritten.
    """
    from pipeline.matcher import archive_existing_plan

    ws_dir = WORKSPACES_DIR / slug
    if not ws_dir.exists():
        raise HTTPException(404, f"Workspace '{slug}' not found")

    clips = body.get("clips", [])
    if not isinstance(clips, list) or not clips:
        raise HTTPException(400, "clips array is required and must be non-empty")

    # Load current lyrics for syncing
    lyrics_file = ws_dir / "state" / "lyrics.json"
    displayable_lyrics: list[dict] = []
    if lyrics_file.exists():
        try:
            raw_lyrics = json.loads(lyrics_file.read_text())
            displayable_lyrics = [
                l for l in raw_lyrics
                if l.get("start_time", 0) != l.get("end_time", 0)
            ]
        except Exception:
            pass

    # Build segments, re-syncing lyric_lines from lyrics.json
    segments = []
    for clip in clips:
        s0 = float(clip.get("song_start", 0))
        s1 = float(clip.get("song_end", 0))
        lyric_lines = [
            l for l in displayable_lyrics
            if l.get("end_time", 0) > s0 and l.get("start_time", 0) < s1
        ] if s1 > s0 else []
        segments.append({
            "scene_index":      int(clip["scene_index"]),
            "lyric_lines":      lyric_lines,
            "scene_trim_start": float(clip.get("scene_trim_start", 0)),
            "scene_trim_end":   float(clip.get("scene_trim_end", -1)),
            "song_start":       s0,
            "song_end":         s1,
        })

    # Determine plan file path (may not exist yet if first save)
    plan_file = next(ws_dir.glob("*_plan.json"), None)
    cfg_file = ws_dir / "workspace.json"
    if not plan_file:
        cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
        slug_str = re.sub(r"[^a-z0-9]+", "_",
                          f"{cfg.get('film','')}_{cfg.get('song','')}".lower()).strip("_")
        plan_file = ws_dir / f"{slug_str}_plan.json"

    # Archive current plan before overwriting
    cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
    slug_str = re.sub(r"[^a-z0-9]+", "_",
                      f"{cfg.get('film','')}_{cfg.get('song','')}".lower()).strip("_")
    archive_existing_plan(ws_dir, slug_str)

    plan_file.write_text(json.dumps(segments, indent=2))
    return {"ok": True, "clips": len(segments)}


@app.get("/api/workspace/{slug}/crop")
def get_crop(slug: str):
    """Return manual crop settings + reference frame URL + video dimensions."""
    import subprocess as _sp
    from config import FFPROBE_BIN
    ws_dir = WORKSPACES_DIR / slug
    cfg_file = ws_dir / "workspace.json"
    if not cfg_file.exists():
        raise HTTPException(404, f"Workspace '{slug}' not found")
    cfg = json.loads(cfg_file.read_text())

    # Probe video dimensions from the stored video path
    video_w, video_h = 1920, 1080
    vp_file = ws_dir / "state" / "video_path.txt"
    if vp_file.exists():
        vp = Path(vp_file.read_text().strip())
        if vp.exists():
            r = _sp.run(
                [FFPROBE_BIN, "-v", "quiet", "-print_format", "json",
                 "-show_entries", "stream=width,height", "-select_streams", "v:0", str(vp)],
                capture_output=True, text=True,
            )
            try:
                streams = json.loads(r.stdout).get("streams", [])
                video_w = int(streams[0]["width"])
                video_h = int(streams[0]["height"])
            except Exception:
                pass

    # Pick a representative reference frame for the crop tool.
    # Priority 1: first scene in the current plan (curated content, guaranteed to show film).
    # Priority 2: a scene from the middle of the sampled frames (avoids opening credits / black).
    frame_url = None
    frames_dir = ws_dir / "frames"
    if frames_dir.exists():
        scene_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
        if scene_dirs:
            # Try to use the scene_index of the first plan clip
            chosen_dir = None
            plan_file = ws_dir / f"{slug}_plan.json"
            if plan_file.exists():
                try:
                    plan = json.loads(plan_file.read_text())
                    if plan:
                        first_scene_idx = plan[0].get("scene_index", 0)
                        target = frames_dir / f"scene_{first_scene_idx:04d}"
                        if target.is_dir():
                            chosen_dir = target
                except Exception:
                    pass
            # Fallback: middle scene (avoids opening-credits / black frames)
            if chosen_dir is None:
                chosen_dir = scene_dirs[len(scene_dirs) // 2]
            frames = sorted(chosen_dir.glob("*.jpg"))
            if frames:
                # Use the middle frame of the chosen scene for best content coverage
                frame_url = f"/api/frame?path={frames[len(frames) // 2]}"

    manual_crop = cfg.get("manual_crop")
    return {
        "top":      manual_crop.get("top", 0) if manual_crop else 0,
        "bottom":   manual_crop.get("bottom", 0) if manual_crop else 0,
        "left":     manual_crop.get("left", 0) if manual_crop else 0,
        "right":    manual_crop.get("right", 0) if manual_crop else 0,
        "active":   bool(manual_crop and any(manual_crop.get(k, 0) for k in ("top", "bottom", "left", "right"))),
        "video_w":  video_w,
        "video_h":  video_h,
        "frame_url": frame_url,
    }


@app.put("/api/workspace/{slug}/crop")
def set_crop(slug: str, body: dict):
    """Save manual crop settings to workspace.json."""
    ws_dir = WORKSPACES_DIR / slug
    cfg_file = ws_dir / "workspace.json"
    if not cfg_file.exists():
        raise HTTPException(404, f"Workspace '{slug}' not found")
    cfg = json.loads(cfg_file.read_text())
    cfg["manual_crop"] = {
        "top":    max(0, int(body.get("top", 0))),
        "bottom": max(0, int(body.get("bottom", 0))),
        "left":   max(0, int(body.get("left", 0))),
        "right":  max(0, int(body.get("right", 0))),
    }
    cfg_file.write_text(json.dumps(cfg, indent=2))
    return {"ok": True}


@app.delete("/api/workspace/{slug}/crop")
def clear_crop(slug: str):
    """Remove manual crop from workspace.json (revert to auto-detection)."""
    ws_dir = WORKSPACES_DIR / slug
    cfg_file = ws_dir / "workspace.json"
    if not cfg_file.exists():
        raise HTTPException(404, f"Workspace '{slug}' not found")
    cfg = json.loads(cfg_file.read_text())
    cfg.pop("manual_crop", None)
    cfg_file.write_text(json.dumps(cfg, indent=2))
    return {"ok": True}


@app.get("/api/fonts")
def list_fonts():
    """Return available subtitle font options for this machine."""
    from config import SUBTITLE_FONTS
    return [{"name": k, "path": v} for k, v in SUBTITLE_FONTS.items()]


@app.post("/api/workspace/{slug}/render")
async def render_stream(slug: str, body: dict = None):
    """Run render and stream output as SSE."""
    body = body or {}
    ws_dir = WORKSPACES_DIR / slug
    cfg_file = ws_dir / "workspace.json"
    if not cfg_file.exists():
        raise HTTPException(404, f"Workspace '{slug}' not found")
    cfg = json.loads(cfg_file.read_text())
    args = ["render", "--film", cfg["film"], "--song", cfg["song"]]
    if cfg.get("director"):
        args.extend(["--director", cfg["director"]])
    if body.get("subtitle_font"):
        args.extend(["--subtitle-font", body["subtitle_font"]])
    if body.get("subtitle_fontsize"):
        args.extend(["--subtitle-fontsize", str(int(body["subtitle_fontsize"]))])
    if body.get("subtitle_italic"):
        args.append("--subtitle-italic")
    return StreamingResponse(_stream_cli(args), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/api/workspace/{slug}/publish")
async def publish_stream(slug: str, body: dict):
    """Run publish and stream output as SSE."""
    ws_dir = WORKSPACES_DIR / slug
    cfg_file = ws_dir / "workspace.json"
    if not cfg_file.exists():
        raise HTTPException(404, f"Workspace '{slug}' not found")
    cfg = json.loads(cfg_file.read_text())
    caption = body.get("caption", "").strip()
    args = ["publish", "--film", cfg["film"], "--song", cfg["song"]]
    if caption:
        args.extend(["--caption", caption])
    return StreamingResponse(_stream_cli(args), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/api/workspace/{slug}/lyrics")
def get_lyrics(slug: str):
    ws_dir = WORKSPACES_DIR / slug
    lyrics_file = ws_dir / "state" / "lyrics.json"
    if not lyrics_file.exists():
        raise HTTPException(404, "No lyrics — run extract-lyrics first")
    return json.loads(lyrics_file.read_text())


@app.put("/api/workspace/{slug}/lyrics")
async def save_lyrics(slug: str, body: dict):
    ws_dir = WORKSPACES_DIR / slug
    if not ws_dir.exists():
        raise HTTPException(404, f"Workspace '{slug}' not found")
    lyrics = body.get("lyrics", [])
    if not isinstance(lyrics, list):
        raise HTTPException(400, "lyrics must be a list")

    lyrics_file = ws_dir / "state" / "lyrics.json"
    lyrics_file.write_text(json.dumps(lyrics, indent=2))

    # Sync lyric_lines in plan.json — editor.py reads exclusively from the plan,
    # so this keeps subtitle timing in sync with manual corrections made here.
    plan_file = next(ws_dir.glob("*_plan.json"), None)
    synced = False
    if plan_file:
        try:
            segments = json.loads(plan_file.read_text())
            displayable = [
                l for l in lyrics
                if l.get("start_time", 0) != l.get("end_time", 0)
            ]
            changed = False
            for seg in segments:
                s0 = seg.get("song_start", -1.0)
                s1 = seg.get("song_end", -1.0)
                if s0 < 0 or s1 < 0:
                    continue  # legacy plan format — skip
                seg["lyric_lines"] = [
                    l for l in displayable
                    if l.get("end_time", 0) > s0 and l.get("start_time", 0) < s1
                ]
                changed = True
            if changed:
                plan_file.write_text(json.dumps(segments, indent=2))
                synced = True
        except Exception as exc:
            print(f"[web_app] plan sync failed: {exc}")

    return {"ok": True, "synced": synced}


@app.get("/api/workspace/{slug}/audio")
def serve_audio(slug: str):
    ws_dir = WORKSPACES_DIR / slug
    audio_path_file = ws_dir / "state" / "audio_path.txt"
    if not audio_path_file.exists():
        raise HTTPException(404, "No audio — run download-audio first")
    audio_path = Path(audio_path_file.read_text().strip())
    if not audio_path.exists():
        raise HTTPException(404, f"Audio file not found: {audio_path}")
    suffix = audio_path.suffix.lower()
    media_types = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".wav": "audio/wav"}
    media_type = media_types.get(suffix, "audio/mpeg")
    return FileResponse(
        str(audio_path),
        media_type=media_type,
        headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache"},
    )


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
