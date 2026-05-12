"""
Transcription pipeline for The Ben and Emil Show.
Downloads audio from YouTube, transcribes with faster-whisper (CUDA), and loads into PostgreSQL.

Usage:
    python transcribe.py                  # Process all new videos
    python transcribe.py --limit 5        # Process only 5 videos
    python transcribe.py --video VIDEO_ID # Process a single video
"""

import os
import sys
import argparse

# Force UTF-8 output on Windows to handle non-ASCII video titles
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import psycopg2
from faster_whisper import WhisperModel

# ── Config ──────────────────────────────────────────────────────────────────
PLAYLIST_ID      = "PL7x4K5eDC-JOeAaRI9p7s7RM2lbRU9GpX"
CHANNEL_SOURCE   = "benandemilshow"
WHISPER_MODEL    = "large-v2"       # large-v2 gives best accuracy; use "medium" if VRAM is tight
DEVICE           = "cuda"
COMPUTE_TYPE     = "float16"        # float16 is optimal for RTX 3060 Ti
AUDIO_DIR        = Path("audio")    # temporary audio storage
DB_ENV_VAR       = "DATABASE_URL_BENANDEMIL"

# ── Database ─────────────────────────────────────────────────────────────────

def get_conn():
    db_url = os.environ.get(DB_ENV_VAR) or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit(f"ERROR: Set {DB_ENV_VAR} or DATABASE_URL in your .env / environment")
    return psycopg2.connect(db_url)


def ensure_schema(conn):
    """Create the quotes table and indexes if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS quotes (
                id               SERIAL PRIMARY KEY,
                video_id         TEXT NOT NULL,
                title            TEXT,
                upload_date      DATE,
                channel_source   TEXT,
                text             TEXT,
                line_number      TEXT,
                timestamp_start  REAL,
                game_name        TEXT,
                fts_doc          TSVECTOR
            );

            CREATE INDEX IF NOT EXISTS idx_quotes_video_id
                ON quotes (video_id);

            CREATE INDEX IF NOT EXISTS idx_quotes_channel_source
                ON quotes (channel_source);

            CREATE INDEX IF NOT EXISTS idx_quotes_upload_date
                ON quotes (upload_date);

            CREATE INDEX IF NOT EXISTS idx_quotes_game_name
                ON quotes (game_name);

            CREATE INDEX IF NOT EXISTS idx_quotes_fts
                ON quotes USING GIN (fts_doc);
        """)
        conn.commit()
    print("Schema ready.")


def already_processed(conn, video_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM quotes WHERE video_id = %s LIMIT 1", (video_id,))
        return cur.fetchone() is not None


def insert_quotes(conn, video_id: str, title: str, upload_date, segments):
    rows = []
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        rows.append((
            video_id,
            title,
            upload_date,
            CHANNEL_SOURCE,
            text,
            str(i + 1),
            seg.start,
            None,  # game_name — not needed for a podcast
        ))

    if not rows:
        return 0

    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO quotes
                (video_id, title, upload_date, channel_source, text, line_number,
                 timestamp_start, game_name, fts_doc)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s,
                 to_tsvector('simple', %s))
        """, [r + (r[4],) for r in rows])   # append text again for fts_doc
        conn.commit()
    return len(rows)


# ── YouTube helpers ──────────────────────────────────────────────────────────

def fetch_video_list(limit: int | None = None) -> list[dict]:
    """Return list of {id, title, upload_date} for all channel videos."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(id)s\t%(title)s\t%(upload_date)s",
        "--no-warnings",
        f"https://www.youtube.com/playlist?list={PLAYLIST_ID}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    videos = []
    stdout = result.stdout or ""
    for line in stdout.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        vid_id = parts[0].strip()
        title  = parts[1].strip() if len(parts) > 1 else ""
        raw_date = parts[2].strip() if len(parts) > 2 else ""
        upload_date = None
        if raw_date and re.match(r"^\d{8}$", raw_date):
            try:
                upload_date = datetime.strptime(raw_date, "%Y%m%d").date()
            except ValueError:
                pass
        videos.append({"id": vid_id, "title": title, "upload_date": upload_date})

    if limit:
        videos = videos[:limit]
    return videos


def fetch_single_video(video_id: str) -> dict:
    cmd = [
        "yt-dlp",
        "--print", "%(id)s\t%(title)s\t%(upload_date)s",
        "--no-warnings",
        "--no-playlist",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = (result.stdout or "").strip()
    parts = line.split("\t", 2)
    title = parts[1].strip() if len(parts) > 1 else ""
    raw_date = parts[2].strip() if len(parts) > 2 else ""
    upload_date = None
    if raw_date and re.match(r"^\d{8}$", raw_date):
        try:
            upload_date = datetime.strptime(raw_date, "%Y%m%d").date()
        except ValueError:
            pass
    return {"id": video_id, "title": title, "upload_date": upload_date}


def download_audio(video_id: str, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--format", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "--output", out_template,
        "--no-warnings",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    audio_path = out_dir / f"{video_id}.wav"
    if audio_path.exists():
        return audio_path
    # yt-dlp might keep another extension
    for f in out_dir.glob(f"{video_id}.*"):
        return f
    print(f"  [WARN] Audio download failed for {video_id}: {result.stderr[:200]}")
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transcribe Ben and Emil Show videos")
    parser.add_argument("--limit",  type=int, help="Max number of videos to process")
    parser.add_argument("--video",  type=str, help="Process a single video ID")
    args = parser.parse_args()

    # Load .env manually (no python-dotenv dependency)
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    conn = get_conn()
    ensure_schema(conn)

    print(f"Loading faster-whisper model '{WHISPER_MODEL}' on {DEVICE} ({COMPUTE_TYPE})…")
    model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Model loaded.")

    if args.video:
        videos = [fetch_single_video(args.video)]
    else:
        print("Fetching video list from channel…")
        videos = fetch_video_list(limit=args.limit)
        print(f"Found {len(videos)} videos.")

    skipped = 0
    processed = 0
    failed = 0

    for i, video in enumerate(videos, 1):
        vid_id = video["id"]
        title  = video["title"]
        print(f"\n[{i}/{len(videos)}] {title} ({vid_id})")

        if already_processed(conn, vid_id):
            print("  Already in DB, skipping.")
            skipped += 1
            continue

        audio_path = download_audio(vid_id, AUDIO_DIR)
        if not audio_path:
            failed += 1
            continue

        try:
            print(f"  Transcribing {audio_path.name}…")
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
            )
            segments = list(segments)  # consume generator
            print(f"  {len(segments)} segments, duration {info.duration:.0f}s")

            n = insert_quotes(conn, vid_id, title, video["upload_date"], segments)
            print(f"  Inserted {n} quotes.")
            processed += 1
        except Exception as e:
            print(f"  ERROR during transcription: {e}")
            failed += 1
        finally:
            try:
                audio_path.unlink()
            except Exception:
                pass

    conn.close()
    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
