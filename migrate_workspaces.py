"""One-time migration: split each workspace into clips/ + songs/ + workspaces/.

Run with:  uv run python migrate_workspaces.py [--dry-run]
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT     = Path(__file__).parent
CLIPS    = ROOT / "clips"
SONGS    = ROOT / "songs"
WKSPS    = ROOT / "workspaces"
DRY_RUN  = "--dry-run" in sys.argv

CLIPS.mkdir(exist_ok=True)
SONGS.mkdir(exist_ok=True)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def move(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"    skip (not found): {src.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if DRY_RUN:
        print(f"    mv  {src.relative_to(ROOT)}  →  {dst.relative_to(ROOT)}")
    else:
        shutil.move(str(src), str(dst))
        print(f"    mv  {src.relative_to(ROOT)}  →  {dst.relative_to(ROOT)}")


def move_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"    skip (not found): {src.relative_to(ROOT)}")
        return
    if DRY_RUN:
        print(f"    mv  {src.relative_to(ROOT)}/  →  {dst.relative_to(ROOT)}/")
    else:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        print(f"    mv  {src.relative_to(ROOT)}/  →  {dst.relative_to(ROOT)}/")


for ws_dir in sorted(WKSPS.iterdir()):
    if not ws_dir.is_dir():
        continue

    cfg_file = ws_dir / "workspace.json"
    if not cfg_file.exists():
        print(f"\n[skip] {ws_dir.name} — no workspace.json")
        continue

    cfg = json.loads(cfg_file.read_text())

    # Already migrated?
    if cfg.get("clip_slug") and cfg.get("song_slug"):
        print(f"\n[ok]   {ws_dir.name} — already migrated")
        continue

    film   = cfg.get("film", "")
    song   = cfg.get("song", "")
    artist = cfg.get("artist", "")

    if not film or not song:
        print(f"\n[skip] {ws_dir.name} — missing film or song in workspace.json")
        continue

    clip_slug = slug(film)
    song_slug = slug(f"{artist}_{song}") if artist else slug(song)

    clip_dir = CLIPS / clip_slug
    song_dir = SONGS / song_slug

    print(f"\n── {ws_dir.name}")
    print(f"   clip: clips/{clip_slug}/")
    print(f"   song: songs/{song_slug}/")

    state = ws_dir / "state"

    # ── Video-related → clips/{clip_slug}/ ─────────────────────────────────
    move(state / "scenes.json",     clip_dir / "scenes.json")
    move(state / "video_path.txt",  clip_dir / "video_path.txt")
    move_tree(ws_dir / "frames",    clip_dir / "frames")

    # Scene-analysis cache → clips/{clip_slug}/cache/
    cache_src = ws_dir / "cache"
    if cache_src.exists():
        for f in cache_src.glob("scene_analysis_*.json"):
            move(f, clip_dir / "cache" / f.name)

    # Video file itself → clips/{clip_slug}/
    for vf in ws_dir.glob("video*.mp4"):
        move(vf, clip_dir / vf.name)

    # ── Audio-related → songs/{song_slug}/ ─────────────────────────────────
    move(state / "lyrics.json",          song_dir / "lyrics.json")
    move(state / "lyrics_readable.txt",  song_dir / "lyrics_readable.txt")
    move(state / "audio_path.txt",       song_dir / "audio_path.txt")

    # Audio file itself → songs/{song_slug}/
    for af in list(ws_dir.glob("*.mp3")) + list(ws_dir.glob("*.m4a")):
        move(af, song_dir / af.name)

    # ── Remaining cache (LLM responses) stays in workspaces/{slug}/cache/ ──
    # Nothing to move — llm_responses_*.json already lives there.

    # ── Remove empty state/ dir ─────────────────────────────────────────────
    if not DRY_RUN and state.exists():
        remaining = list(state.iterdir())
        if not remaining:
            state.rmdir()
            print(f"    rm  {state.relative_to(ROOT)}/  (empty)")
        else:
            print(f"    note: {state.relative_to(ROOT)}/ still has: {[f.name for f in remaining]}")

    # ── Update workspace.json ───────────────────────────────────────────────
    cfg["clip_slug"] = clip_slug
    cfg["song_slug"] = song_slug

    if not DRY_RUN:
        cfg_file.write_text(json.dumps(cfg, indent=2))
        print(f"    updated workspace.json  (clip_slug={clip_slug!r}, song_slug={song_slug!r})")
    else:
        print(f"    [dry] workspace.json += clip_slug={clip_slug!r}, song_slug={song_slug!r}")

    # ── Update stored audio/video paths if they point inside the workspace ──
    # The path files contain the absolute path to the actual media file.
    # If the media file moved, update the reference.
    for path_file, new_dir in [
        (clip_dir / "video_path.txt", clip_dir),
        (song_dir  / "audio_path.txt", song_dir),
    ]:
        if not DRY_RUN and path_file.exists():
            old_path = Path(path_file.read_text().strip())
            if old_path.parent == ws_dir:
                # Media file was in workspace root — it has moved to new_dir
                new_path = new_dir / old_path.name
                if new_path.exists():
                    path_file.write_text(str(new_path))
                    print(f"    updated {path_file.relative_to(ROOT)}: {new_path.relative_to(ROOT)}")

print("\nMigration complete." if not DRY_RUN else "\n[dry run] No files were changed.")
