# python realtime_pa_ingest.py init_db
# python realtime_pa_ingest.py ingest_pa
# python realtime_pa_ingest.py ingest_pa --limit 1
# python realtime_pa_ingest.py ingest_pa --since "2025-02-01T10:00:00Z"
# python realtime_pa_ingest.py watch --interval 60 --limit 5

import os                                   # os: interact with environment variables and file system
import io                                   # io: kept for completeness (in-memory buffers)
import time                                 # time: sleep and timing utilities used in watch loop
import argparse                             # argparse: parse command-line arguments
import tempfile                             # tempfile: create temporary files for downloaded audio
from typing import Optional, List          # typing: Optional and List type hints for clarity
from datetime import datetime, timezone    # datetime: timezone-aware timestamps
from urllib.parse import urlparse, quote   # urlparse/quote: parse URLs and safely quote path segments
import mimetypes                           # mimetypes: useful utilities to guess mime types (not heavily used)
import math                                # math: numeric helpers left available if needed
import signal                               # signal: handle SIGINT/SIGTERM for graceful shutdown
import sys                                  # sys: for exiting with status and stderr
import traceback                            # traceback: optional detailed stack traces for debugging

# SQLAlchemy imports for database interaction
from sqlalchemy import create_engine, text  # create_engine/text: create DB engine and execute safe SQL
from sqlalchemy.engine import Engine        # Engine typing alias for clarity
from sqlalchemy.exc import SQLAlchemyError  # SQLAlchemyError: catch DB related exceptions

# Try to load optional .env file using python-dotenv
try:
    from dotenv import load_dotenv          # import load_dotenv to bring .env into environment
    load_dotenv()                           # call load_dotenv so .env variables are available to os.environ
    print(" Environment: .env loaded (if present)")  # print confirmation if .env loaded
except Exception:
    print(" Environment: python-dotenv not found, using system env vars")  # fallback if python-dotenv isn't installed

# Try to import requests for HTTP downloads; script requires it for network access
try:
    import requests                         # requests: used for HTTP(S) downloads of recording files
except Exception:
    requests = None                         # if import fails, set to None and raise later when needed

# Try to import modern OpenAI SDK, otherwise fallback to legacy openai package
try:
    from openai import OpenAI               # attempt to import modern OpenAI client class
    _HAS_NEW_OPENAI = True                  # flag new SDK available
    print(" OpenAI: using new SDK (OpenAI)")
except Exception:
    _HAS_NEW_OPENAI = False                 # mark that modern SDK isn't present
    try:
        import openai  # type: ignore
        print(" OpenAI: using legacy openai package (import succeeded)")
    except Exception:
        print(" OpenAI SDK not found; ASR will fail if attempted (install openai package)")

# ========================== Configuration ==========================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://user:pass@host:5432/dbname"
)                                           # DATABASE_URL: read from env or use fallback

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # OPENAI_API_KEY: must be set to run ASR
ASR_MODEL = os.getenv("ASR_MODEL", "whisper-1")   # ASR_MODEL: default Whisper model for transcription

DEFAULT_SINCE_ISO = os.getenv("PA_INGEST_DEFAULT_SINCE", "2025-11-17T13:00:00.705Z")  # fallback since timestamp
PA_RECORDING_HTTP_PREFIX = os.getenv("PA_RECORDING_HTTP_PREFIX", "").rstrip("/")  # optional prefix for path-only DB values

ALLOWED_EXTS: List[str] = [
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".oga", ".webm", ".mp4", ".aac", ".wma"
]                                           # list of allowed file extensions used as quick guard

ALLOWED_CONTENT_TYPE_SUBSTRINGS = [
    "audio/", "mpeg", "wav", "ogg", "webm", "mpeg3", "mpeg4", "mp3", "x-wav", "x-m4a", "octet-stream"
]                                           # substrings to validate Content-Type header

MIN_AUDIO_BYTES = 2 * 1024                    # MIN_AUDIO_BYTES: small threshold (2 KB) to avoid trivial downloads
DEFAULT_POLL_INTERVAL = 30                     # default polling interval in seconds when using watch

# Module-level flag to request graceful shutdown from signal handler
_SHUTDOWN_REQUESTED = False

def _signal_handler(signum, frame):
    """Signal handler to set shutdown flag when SIGINT or SIGTERM is received."""
    global _SHUTDOWN_REQUESTED                     # reference module-level shutdown flag
    print(f"\n Signal received ({signum}), requesting graceful shutdown...")  # log signal receipt
    _SHUTDOWN_REQUESTED = True                     # set flag so watch loop can exit cleanly

# Register signal handlers for graceful termination
signal.signal(signal.SIGINT, _signal_handler)     # handle Ctrl+C (SIGINT)
signal.signal(signal.SIGTERM, _signal_handler)    # handle termination (SIGTERM)

# ========================== DB Utilities ==========================
def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine using DATABASE_URL."""
    return create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=5)  # engine with pre-ping to avoid stale connections


def run_sql(engine: Engine, sql: str, params: Optional[dict] = None) -> None:
    """Execute a non-SELECT SQL statement using a transaction context."""
    with engine.begin() as conn:                    # open a transaction-scoped connection
        conn.execute(text(sql), params or {})       # execute provided SQL with bound parameters


def fetch_all(engine: Engine, sql: str, params: Optional[dict] = None) -> List[dict]:
    """Execute a SELECT query and return rows as list of dicts for easy consumption."""
    with engine.begin() as conn:                    # open a transaction-scoped connection
        result = conn.execute(text(sql), params or {})  # execute select query
        return [dict(r._mapping) for r in result.fetchall()]  # convert SQLAlchemy Row -> dict

# -------------------------- Schema: pa_recordings --------------------------
DDL_PA_RECORDINGS = """
CREATE TABLE IF NOT EXISTS public.pa_recordings (
    id BIGSERIAL PRIMARY KEY,
    pa_recording_url TEXT UNIQUE NOT NULL,
    created TIMESTAMPTZ NOT NULL,
    pa_en_transcribe TEXT,
    call_type TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_pa_recordings_created ON public.pa_recordings (created);
"""                                         # SQL DDL to ensure table exists and index for created


def init_db(engine: Optional[Engine] = None):
    """Ensure the pa_recordings table and index exist (idempotent operation)."""
    engine_to_use = engine if engine is not None else get_engine()  # allow passing engine for reuse
    try:
        run_sql(engine_to_use, DDL_PA_RECORDINGS)   # run the DDL to create table/index if missing
        print(" pa_recordings table ensured/created successfully.")
    except SQLAlchemyError as e:
        print(f" Database initialization failed: {e}")  # log DB initialization error
        raise

# ====================== OpenAI helper ======================
def _get_openai_client():
    """Return an OpenAI client (modern SDK) or configure legacy module-level API key.

    Raises RuntimeError when OPENAI_API_KEY is not set.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")  # require API key for ASR
    if _HAS_NEW_OPENAI:
        return OpenAI(api_key=OPENAI_API_KEY)      # instantiate and return modern OpenAI client
    else:
        import openai  # type: ignore
        openai.api_key = OPENAI_API_KEY            # configure legacy package's api key
        return None                                # legacy usage uses module-level functions


def transcribe_audio_file(local_file_path: str, prefer_translate: bool = True) -> str:
    """Transcribe or translate a local audio file using OpenAI audio endpoints.

    - prefer_translate=True uses translations endpoint (better for non-English)
    - Supports new and legacy SDK shapes
    - Returns a string containing the transcription
    """
    client = _get_openai_client()                  # get client or configure legacy module
    try:
        if _HAS_NEW_OPENAI:
            with open(local_file_path, "rb") as f:  # open file in binary mode for upload
                if prefer_translate:
                    resp = client.audio.translations.create(
                        model=ASR_MODEL,
                        file=f,
                        response_format="text",
                        temperature=0.0,
                    )
                else:
                    resp = client.audio.transcriptions.create(
                        model=ASR_MODEL,
                        file=f,
                        response_format="text",
                        temperature=0.0,
                    )
            return resp.strip() if isinstance(resp, str) else str(resp).strip()  # normalize to stripped string
        else:
            import openai  # type: ignore
            with open(local_file_path, "rb") as f:
                if prefer_translate:
                    resp = openai.Audio.translations.create(  # type: ignore
                        model=ASR_MODEL,
                        file=f,
                        response_format="text",
                        temperature=0.0,
                    )
                else:
                    resp = openai.Audio.transcriptions.create(  # type: ignore
                        model=ASR_MODEL,
                        file=f,
                        response_format="text",
                        temperature=0.0,
                    )
            return resp.strip() if isinstance(resp, str) else str(resp).strip()
    except Exception as e:
        print(f"[ASR] Transcription error for {local_file_path}: {e}")  # log ASR-level errors
        raise

# ====================== Helpers for recording_url processing ======================
def file_ext_from_url(url: str) -> str:
    """Extract file extension from URL path or return default .mp3 when missing."""
    parsed = urlparse(url)                         # parse URL to access path component
    _, ext = os.path.splitext(parsed.path)        # split extension from path
    return ext if ext else ".mp3"               # fallback to .mp3 when extension absent


def is_allowed_extension_from_url(url: str) -> bool:
    """Return True if URL extension exists in allowed extensions list."""
    ext = file_ext_from_url(url).lower()          # normalize to lowercase for comparison
    return ext in ALLOWED_EXTS


def content_type_looks_like_audio(content_type: Optional[str]) -> bool:
    """Perform substring checks to decide if Content-Type header looks audio-like."""
    if not content_type:
        return False
    ct = content_type.lower()
    for sub in ALLOWED_CONTENT_TYPE_SUBSTRINGS:
        if sub in ct:
            return True
    return False

# ====================== Upsert into pa_recordings ======================
def upsert_pa_recording(engine: Engine, recording_url: str, created: datetime, transcript: str, call_type: str, status: str = "pending") -> None:
    """Insert or update a pa_recordings row keyed by pa_recording_url using ON CONFLICT."""
    upsert_sql = """
    INSERT INTO public.pa_recordings (pa_recording_url, created, pa_en_transcribe, call_type, status)
    VALUES (:url, :created, :transcript, :call_type, :status)
    ON CONFLICT (pa_recording_url) DO UPDATE SET
      pa_en_transcribe = EXCLUDED.pa_en_transcribe,
      created = EXCLUDED.created,
      call_type = EXCLUDED.call_type,
      status = EXCLUDED.status;
    """
    run_sql(engine, upsert_sql, {
        "url": recording_url,
        "created": created,
        "transcript": transcript,
        "call_type": call_type,
        "status": status
    })

# ====================== High-water-mark helper ======================
def max_pa_created(engine: Engine) -> Optional[str]:
    """Return ISO string for maximum created timestamp stored in pa_recordings or None."""
    rows = fetch_all(engine, "SELECT max(created) AS mx FROM public.pa_recordings")  # query max(created)
    if not rows:
        return None
    mx = rows[0].get("mx")
    if mx is None:
        return None
    return mx.isoformat()                         # convert datetime to ISO string

# ====================== Fetch source rows from outbound_inbound_calling_details ======================
def fetch_oicd_rows(engine: Engine, since_iso: Optional[str], limit: Optional[int] = None) -> List[dict]:
    """Fetch rows from outbound_inbound_calling_details where created > :since ordered ascending by created."""
    if since_iso is None:
        hw = max_pa_created(engine)                   # compute high-water-mark from pa_recordings
        since_param = hw if hw else DEFAULT_SINCE_ISO # fallback to DEFAULT_SINCE_ISO when no high-water-mark
    else:
        since_param = since_iso

    base_sql = "SELECT * FROM public.outbound_inbound_calling_details oicd WHERE oicd.created > :since ORDER BY oicd.created ASC"
    if limit:
        base_sql += " LIMIT :limit"                  # append LIMIT if provided to control per-iteration rows
        params = {"since": since_param, "limit": limit}
    else:
        params = {"since": since_param}

    return fetch_all(engine, base_sql, params)        # return list of dict rows

# ====================== Core ingestion flow (HTTP-only, prefix support) ======================
def ingest_pa_http(since_iso: Optional[str] = None, limit: Optional[int] = None, prefer_translate: bool = True, engine: Optional[Engine] = None) -> dict:
    """Process new rows: download audio via HTTP, transcribe via OpenAI, upsert into pa_recordings.

    Returns a summary dict with counts for the iteration.
    """
    if requests is None:
        raise RuntimeError("The 'requests' library is required. Install via `pip install requests`.")  # require requests module

    engine_in_use = engine if engine is not None else get_engine()  # allow passing a reusable engine
    init_db(engine_in_use)                                         # ensure pa_recordings exists before processing

    since_param_for_log = since_iso if since_iso is not None else "(auto from pa_recordings.max(created) or default)"
    print(f" Determining source rows using since: {since_param_for_log}")

    rows = fetch_oicd_rows(engine_in_use, since_iso, limit=limit)  # fetch candidate source rows
    print(f" Found {len(rows)} candidate rows.")

    processed = 0
    succeeded = 0
    failed = 0
    skipped = 0

    for row in rows:
        processed += 1
        try:
            raw_val = row.get("recording_url") or row.get("recordingUrl") or row.get("recording") or row.get("recordingLink")  # tolerate multiple column names
            created = row.get("created")
            call_type = row.get("call_type") or row.get("calltype") or row.get("direction") or row.get("call_type_io") or "unknown"
            source_id = row.get("id")

            raw_val = "" if raw_val is None else str(raw_val).strip()  # normalize to trimmed string
            print(f"\n[{processed}] Source id={source_id} Original DB value: {raw_val!r}")

            if not raw_val:
                print(f"  Skipping id={source_id} — recording value is empty/whitespace")
                skipped += 1
                continue

            parsed_initial = urlparse(raw_val)                     # parse initial value to check for scheme
            recording_url = raw_val
            if (not parsed_initial.scheme) and PA_RECORDING_HTTP_PREFIX:
                quoted_path = quote(raw_val.lstrip("/"))        # safely quote path when joining
                recording_url = PA_RECORDING_HTTP_PREFIX + "/" + quoted_path
                print(f"  Rewrote DB path to full URL using PA_RECORDING_HTTP_PREFIX -> {recording_url}")

            parsed = urlparse(recording_url)                       # parse final recording_url

            if parsed.path.endswith("/"):
                print(f"  Skipping id={source_id} — URL looks like a directory (no file): {recording_url}")
                skipped += 1
                try:
                    upsert_pa_recording(engine_in_use, recording_url, created, transcript="", call_type=call_type, status="pending")
                    print(f"  Upserted pending record for id={source_id} (directory-like path).")
                except Exception as e:
                    print(f"  Failed to upsert pending record for id={source_id}: {e}")
                continue

            if not is_allowed_extension_from_url(recording_url):
                print(f"  Warning id={source_id}: file extension not in allowed list: {file_ext_from_url(recording_url)}; attempting download but will validate content-type.")

            if parsed.scheme not in ("http", "https"):
                print(f"  Skipping id={source_id} — unsupported URL scheme: {parsed.scheme}")
                skipped += 1
                continue

            local_tmp_path = None
            try:
                print(f"  Downloading id={source_id}: {recording_url}")
                resp = requests.get(recording_url, stream=True, timeout=60)  # stream to avoid large memory usage
                resp.raise_for_status()                                       # raise for HTTP errors (4xx/5xx)
            except Exception as e:
                print(f"  HTTP download failed for id={source_id} url={recording_url}: {e}")
                failed += 1
                try:
                    upsert_pa_recording(engine_in_use, recording_url, created, transcript="", call_type=call_type, status="pending")
                    print(f"  Upserted pending row for id={source_id} after download failure.")
                except Exception as e2:
                    print(f"  Failed to upsert pending row for id={source_id} after download failure: {e2}")
                continue

            content_type = resp.headers.get("content-type", "")
            content_length = resp.headers.get("content-length")
            if not content_type_looks_like_audio(content_type):
                print(f"  Downloaded resource Content-Type not audio-like for id={source_id}: {content_type!r}")
                try:
                    _ = resp.content  # read body to close connection
                except Exception:
                    pass
                failed += 1
                try:
                    upsert_pa_recording(engine_in_use, recording_url, created, transcript="", call_type=call_type, status="pending")
                    print(f"  Upserted pending row for id={source_id} due to non-audio Content-Type.")
                except Exception as e2:
                    print(f"  Failed to upsert pending row for id={source_id}: {e2}")
                continue

            try:
                ext = file_ext_from_url(recording_url)
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="pa_audio_") as tmp:
                    written = 0
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            tmp.write(chunk)
                            written += len(chunk)
                    local_tmp_path = tmp.name
                if written < MIN_AUDIO_BYTES:
                    print(f"  Downloaded file too small for id={source_id} (bytes={written}), skipping ASR.")
                    try:
                        if local_tmp_path and os.path.exists(local_tmp_path):
                            os.remove(local_tmp_path)
                    except Exception:
                        pass
                    failed += 1
                    try:
                        upsert_pa_recording(engine_in_use, recording_url, created, transcript="", call_type=call_type, status="pending")
                        print(f"  Upserted pending row for id={source_id} due to tiny download.")
                    except Exception as e2:
                        print(f"  Failed to upsert pending row for id={source_id}: {e2}")
                    continue
            except Exception as e:
                print(f"  Failed while saving response to temp file for id={source_id}: {e}")
                failed += 1
                try:
                    if local_tmp_path and os.path.exists(local_tmp_path):
                        os.remove(local_tmp_path)
                except Exception:
                    pass
                try:
                    upsert_pa_recording(engine_in_use, recording_url, created, transcript="", call_type=call_type, status="pending")
                    print(f"  Upserted pending row for id={source_id} after temp-file write failure.")
                except Exception as e2:
                    print(f"  Failed to upsert pending row for id={source_id}: {e2}")
                continue

            try:
                print(f"  Transcribing id={source_id} file={local_tmp_path} (prefer_translate={prefer_translate})")
                transcript = transcribe_audio_file(local_tmp_path, prefer_translate=prefer_translate)
                print(f"  Transcription length for id={source_id}: {len(transcript)} characters")
                upsert_pa_recording(engine_in_use, recording_url, created, transcript, call_type, status="pending")
                succeeded += 1
                print(f"  Upserted pending transcript for id={source_id} url={recording_url}")
            except Exception as e:
                print(f"  Transcription/upsert failed for id={source_id} url={recording_url}: {e}")
                failed += 1
                try:
                    upsert_pa_recording(engine_in_use, recording_url, created, transcript="", call_type=call_type, status="pending")
                    print(f"  Upserted pending row for id={source_id} after ASR failure.")
                except Exception as e2:
                    print(f"  Failed to upsert pending row for id={source_id} after ASR failure: {e2}")
            finally:
                try:
                    if local_tmp_path and os.path.exists(local_tmp_path):
                        os.remove(local_tmp_path)
                except Exception as cleanup_err:
                    print(f"  Temp cleanup warning for id={source_id}: {cleanup_err}")

        except Exception as outer:
            print(f"[{processed}] Unexpected error while processing source id={row.get('id')}: {outer}")
            failed += 1
            try:
                url_for_upsert = (row.get("recording_url") or "").strip()
                if not url_for_upsert and PA_RECORDING_HTTP_PREFIX:
                    url_for_upsert = PA_RECORDING_HTTP_PREFIX
                upsert_pa_recording(engine_in_use, url_for_upsert, row.get("created") or datetime.now(timezone.utc), transcript="", call_type=row.get("call_type") or "unknown", status="pending")
                print(f"  Upserted fallback pending row for id={row.get('id')} after unexpected error.")
            except Exception as e2:
                print(f"  Failed fallback upsert after unexpected error for id={row.get('id')}: {e2}")
            continue

    print("\n" + "=" * 60)
    print("PA INGESTION ITERATION SUMMARY")
    print("=" * 60)
    print(f" Candidates fetched: {len(rows)}")
    print(f" Processed attempts: {processed}")
    print(f" Successful (upserted transcript): {succeeded}")
    print(f" Failed: {failed}")
    print(f" Skipped: {skipped}")
    print("=" * 60)

    return {
        "candidates": len(rows),
        "processed": processed,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped
    }

# ====================== Realtime/watch mode ======================
def watch_loop(interval: int = DEFAULT_POLL_INTERVAL, since_iso: Optional[str] = None, limit: Optional[int] = None, prefer_translate: bool = True):
    """Continuously poll for new rows every `interval` seconds and process them using ingest_pa_http.

    The `limit` parameter controls how many rows are processed per iteration (useful to cap workload).
    """
    engine = get_engine()                          # create a single reusable DB engine
    init_db(engine)                                # ensure schema exists before entering loop

    iteration = 0
    print(f" Entering watch loop: polling every {interval} seconds. Press Ctrl+C to stop.")

    global _SHUTDOWN_REQUESTED
    while not _SHUTDOWN_REQUESTED:
        iteration += 1
        print(f"\n--- WATCH ITERATION {iteration} @ {datetime.now(timezone.utc).isoformat()} ---")
        try:
            summary = ingest_pa_http(since_iso=since_iso, limit=limit, prefer_translate=prefer_translate, engine=engine)  # process up to `limit` rows
        except Exception as e:
            print(f" Watch iteration error: {e}")
            traceback.print_exc()
        if _SHUTDOWN_REQUESTED:
            print(" Shutdown requested; breaking watch loop.")
            break
        slept = 0.0
        while slept < interval and not _SHUTDOWN_REQUESTED:
            time.sleep(1.0)                           # sleep in 1s increments to respond quickly to signals
            slept += 1.0
    print(" Watch loop exited cleanly. Goodbye.")

# ============================= CLI =============================
def main():
    """CLI parsing and command dispatch for init_db, ingest_pa, and watch commands."""
    parser = argparse.ArgumentParser(description="PA Call Analyzer (HTTP-only) - ingest & transcribe recordings into pa_recordings (supports watch mode with --limit)")
    subs = parser.add_subparsers(dest="command", help="commands")

    subs.add_parser("init_db", help="Create/ensure public.pa_recordings table")

    p = subs.add_parser("ingest_pa", help="Fetch rows and transcribe recording_url via HTTP(S) for a single pass")
    p.add_argument("--since", type=str, default=None, help="ISO timestamp boundary (default: auto high-water-mark from pa_recordings or fallback)")
    p.add_argument("--limit", type=int, default=None, help="Optional LIMIT for number of rows to process")
    p.add_argument("--no-translate", action="store_true", help="Use transcription (not translation) endpoint")
    p.add_argument("--prefer-translate", action="store_true", help="Prefer translation endpoint (overrides --no-translate)")

    w = subs.add_parser("watch", help="Run ingestion in realtime polling mode (continuous loop)")
    w.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL, help=f"Seconds between polling iterations (default: {DEFAULT_POLL_INTERVAL})")
    w.add_argument("--since", type=str, default=None, help="Optional fixed ISO timestamp boundary to always use (if not provided, high-water-mark is auto-computed each iteration)")
    w.add_argument("--limit", type=int, default=None, help="Optional per-iteration LIMIT (e.g., --limit 5) to cap rows processed per poll")
    w.add_argument("--no-translate", action="store_true", help="Use transcription (not translation) endpoint")
    w.add_argument("--prefer-translate", action="store_true", help="Prefer translation endpoint (overrides --no-translate)")

    args = parser.parse_args()

    if args.command == "init_db":
        init_db()
        print(" Database init complete.")
        return

    if args.command == "ingest_pa":
        if not OPENAI_API_KEY:
            raise SystemExit("CRITICAL: OPENAI_API_KEY environment variable must be set.")
        if requests is None:
            raise SystemExit("CRITICAL: 'requests' library not installed. Install with `pip install requests`.")

        if args.prefer_translate:
            prefer_translate = True
        elif args.no_translate:
            prefer_translate = False
        else:
            prefer_translate = True

        if not PA_RECORDING_HTTP_PREFIX:
            print(" WARNING: PA_RECORDING_HTTP_PREFIX is not set. If DB stores only path/keys (not full URLs), downloads may fail.")

        ingest_pa_http(since_iso=args.since, limit=args.limit, prefer_translate=prefer_translate)
        return

    if args.command == "watch":
        if requests is None:
            raise SystemExit("CRITICAL: 'requests' library not installed. Install with `pip install requests`.")

        if args.prefer_translate:
            prefer_translate = True
        elif args.no_translate:
            prefer_translate = False
        else:
            prefer_translate = True

        if not PA_RECORDING_HTTP_PREFIX:
            print(" WARNING: PA_RECORDING_HTTP_PREFIX is not set. If DB stores only path/keys (not full URLs), downloads may fail.")

        try:
            watch_loop(interval=args.interval, since_iso=args.since, limit=args.limit, prefer_translate=prefer_translate)
        except Exception as e:
            print(f" Fatal error in watch loop: {e}")
            traceback.print_exc()
            raise SystemExit(1)
        return

    parser.print_help()
    raise SystemExit("No valid command specified.")

# ----------------------------- Entry -----------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Operation cancelled by user (KeyboardInterrupt).")
        raise SystemExit(1)
    except Exception as e:
        print(f" Fatal error in main: {e}")
        traceback.print_exc()
        raise SystemExit(1)
