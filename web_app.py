"""FastAPI web app for song-video-maker — pipeline UI + plan editor."""
import asyncio
import json
import os
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
