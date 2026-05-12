"""
Transcription pipeline for Ben and Emil Show BONUS episodes via RSS feed.
Downloads audio directly from the RSS enclosure URLs, transcribes with
faster-whisper (CUDA), and loads into PostgreSQL.

Usage:
    python transcribe_rss.py                  # Process all new bonus episodes
    python transcribe_rss.py --limit 3        # Process only 3 episodes
    python transcribe_rss.py --all            # Include main BAES episodes too
"""

import os
import sys
import argparse
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from email.utils import parsedate_to_datetime

import psycopg2
from faster_whisper import WhisperModel

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env before reading config values
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

# ── Config ───────────────────────────────────────────────────────────────────
RSS_URL        = os.environ.get("BONUS_RSS_URL", "")
CHANNEL_SOURCE = "benandemilshow_bonus"
WHISPER_MODEL  = "large-v2"
DEVICE         = "cuda"
COMPUTE_TYPE   = "float16"
AUDIO_DIR      = Path("audio")
DB_ENV_VAR     = "DATABASE_URL_BENANDEMIL"

# ── Database ──────────────────────────────────────────────────────────────────

def get_conn():
    db_url = os.environ.get(DB_ENV_VAR) or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit(f"ERROR: Set {DB_ENV_VAR} or DATABASE_URL in your .env / environment")
    return psycopg2.connect(db_url)


def ensure_schema(conn):
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
            CREATE INDEX IF NOT EXISTS idx_quotes_video_id      ON quotes (video_id);
            CREATE INDEX IF NOT EXISTS idx_quotes_channel_source ON quotes (channel_source);
            CREATE INDEX IF NOT EXISTS idx_quotes_upload_date    ON quotes (upload_date);
            CREATE INDEX IF NOT EXISTS idx_quotes_fts            ON quotes USING GIN (fts_doc);
        """)
        conn.commit()
    print("Schema ready.")


def already_processed(conn, video_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM quotes WHERE video_id = %s LIMIT 1", (video_id,))
        return cur.fetchone() is not None


def insert_quotes(conn, video_id, title, upload_date, segments):
    rows = [
        (video_id, title, upload_date, CHANNEL_SOURCE,
         seg.text.strip(), str(i + 1), seg.start, None)
        for i, seg in enumerate(segments) if seg.text.strip()
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO quotes
                (video_id, title, upload_date, channel_source, text,
                 line_number, timestamp_start, game_name, fts_doc)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
        """, [r + (r[4],) for r in rows])
        conn.commit()
    return len(rows)


# ── RSS parsing ───────────────────────────────────────────────────────────────

def fetch_episodes(bonus_only: bool = True) -> list[dict]:
    print(f"Fetching RSS feed…")
    req = urllib.request.Request(RSS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_bytes = resp.read()

    root = ET.fromstring(xml_bytes)
    channel = root.find("channel")
    ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}

    episodes = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()

        # Filter: skip main BAES episodes if bonus_only
        if bonus_only and not title.lower().startswith("bonus"):
            continue

        guid  = (item.findtext("guid") or "").strip()
        enclosure = item.find("enclosure")
        audio_url = enclosure.get("url") if enclosure is not None else None
        if not audio_url:
            continue

        pub_date_raw = item.findtext("pubDate") or ""
        upload_date = None
        try:
            upload_date = parsedate_to_datetime(pub_date_raw).date()
        except Exception:
            pass

        episodes.append({
            "id":         f"rss_{guid}",   # unique DB key
            "title":      title,
            "upload_date": upload_date,
            "audio_url":  audio_url,
        })

    return episodes


# ── Audio download ────────────────────────────────────────────────────────────

def download_audio(episode: dict, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id   = re.sub(r"[^a-zA-Z0-9_-]", "_", episode["id"])
    out_path  = out_dir / f"{safe_id}.m4a"

    if out_path.exists():
        return out_path

    url = episode["audio_url"]
    print(f"  Downloading audio…")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
            while chunk := resp.read(1 << 16):
                f.write(chunk)
        return out_path
    except Exception as e:
        print(f"  [WARN] Download failed: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transcribe Ben and Emil bonus episodes from RSS")
    parser.add_argument("--limit", type=int, help="Max episodes to process")
    parser.add_argument("--all",   action="store_true", help="Include main BAES episodes too")
    args = parser.parse_args()

    # Load .env
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    if not RSS_URL:
        sys.exit("ERROR: Set BONUS_RSS_URL in your .env file (your private podcast RSS URL)")

    conn = get_conn()
    ensure_schema(conn)

    print(f"Loading faster-whisper model '{WHISPER_MODEL}' on {DEVICE} ({COMPUTE_TYPE})…")
    model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Model loaded.")

    episodes = fetch_episodes(bonus_only=not args.all)
    if args.limit:
        episodes = episodes[:args.limit]
    print(f"Found {len(episodes)} {'bonus ' if not args.all else ''}episodes.")

    skipped = processed = failed = 0

    for i, ep in enumerate(episodes, 1):
        print(f"\n[{i}/{len(episodes)}] {ep['title']} ({ep['id']})")

        if already_processed(conn, ep["id"]):
            print("  Already in DB, skipping.")
            skipped += 1
            continue

        audio_path = download_audio(ep, AUDIO_DIR)
        if not audio_path:
            failed += 1
            continue

        try:
            print(f"  Transcribing…")
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
            )
            segments = list(segments)
            print(f"  {len(segments)} segments, duration {info.duration:.0f}s")

            n = insert_quotes(conn, ep["id"], ep["title"], ep["upload_date"], segments)
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
