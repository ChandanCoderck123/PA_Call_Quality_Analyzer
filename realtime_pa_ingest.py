# python realtime_pa_ingest.py init_db
# python realtime_pa_ingest.py ingest_pa --limit 1

# realtime_pa_ingest.py - HTTP-only PA Call Analyzer ingest/transcribe/upsert script
# Fully commented file — each line has an explanatory comment of what it does.
# # Initialize database (one-time)
# python realtime_pa_ingest.py init_db

# # Test with one recording
# python realtime_pa_ingest.py ingest_pa --limit 1

# # Process all new recordings
# python realtime_pa_ingest.py ingest_pa

"""
PA Call Analyzer - HTTP ingest/transcribe/upsert
- Ensures public.pa_recordings table exists
- Fetches new rows from public.outbound_inbound_calling_details using a high-water-mark
- Rewrites DB path-only values using PA_RECORDING_HTTP_PREFIX when needed
- Downloads audio via HTTP(S), validates response is audio, transcribes using OpenAI
- Upserts transcripts into public.pa_recordings with status='pending' and created = source.created
"""

# ------------------------ Standard Library Imports ------------------------
import os                                   # read environment variables and file operations
import io                                   # in-memory buffers (kept for completeness)
import time                                 # backoff/sleep (not heavily used)
import argparse                             # parse CLI arguments
import tempfile                             # create temporary files for downloaded audio
from typing import Optional, List          # type hints for Optional and List
from datetime import datetime, timezone    # parse and handle timestamps/timezones
from urllib.parse import urlparse, quote   # parse URLs and safely quote path segments
import mimetypes                           # guess mime types by extension
import math                                # used for file size checks (optional)

# ------------------------ SQLAlchemy Imports ------------------------
from sqlalchemy import create_engine, text  # create DB engine and execute safe SQL text
from sqlalchemy.engine import Engine        # Engine type
from sqlalchemy.exc import SQLAlchemyError  # catch SQLAlchemy errors

# ------------------------ Optional .env loader ------------------------
try:
    from dotenv import load_dotenv          # optional .env loader
    load_dotenv()                           # load .env into environment variables if present
    print(" Environment: .env loaded (if present)")
except Exception:
    print(" Environment: python-dotenv not found, using system env vars")

# ------------------------ HTTP download lib (requests) ------------------------
try:
    import requests                         # requests for HTTP(S) downloads
except Exception:
    requests = None                         # set to None when requests is not available

# ------------------------ OpenAI SDK (New + Legacy) ------------------------
try:
    from openai import OpenAI               # attempt to use modern OpenAI SDK
    _HAS_NEW_OPENAI = True                  # mark that new SDK is available
    print(" OpenAI: using new SDK (OpenAI)")
except Exception:
    _HAS_NEW_OPENAI = False                 # fallback to legacy package
    import openai  # type: ignore
    print(" OpenAI: using legacy openai package")

# ========================== Configuration ==========================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://user:pass@host:5432/dbname"
)                                           # fallback DB URL (override in production)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # OpenAI API key required for ASR calls
ASR_MODEL = os.getenv("ASR_MODEL", "whisper-1")   # default ASR model (can be overridden)

# default ISO timestamp if none provided (kept from your prior setting)
DEFAULT_SINCE_ISO = os.getenv("PA_INGEST_DEFAULT_SINCE", "2025-11-17T13:00:00.705Z")

# Optional HTTP prefix to prepend to DB path values (no trailing slash)
PA_RECORDING_HTTP_PREFIX = os.getenv("PA_RECORDING_HTTP_PREFIX", "").rstrip("/")

# quick guard of allowed audio file extensions
ALLOWED_EXTS: List[str] = [
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".oga", ".webm", ".mp4", ".aac", ".wma"
]

# Acceptable audio-ish content type substrings we will allow to pass to ASR
ALLOWED_CONTENT_TYPE_SUBSTRINGS = [
    "audio/", "mpeg", "wav", "ogg", "webm", "mpeg", "mpeg3", "mpeg4", "mp3", "x-wav", "x-m4a", "octet-stream"
]

# Minimum file size in bytes below which we consider the download invalid for ASR
MIN_AUDIO_BYTES = 2 * 1024                    # 2 KB minimal sanity check (raise if needed)

# ========================== DB Utilities ==========================
def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine (reused across the run)."""
    return create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=5)

def run_sql(engine: Engine, sql: str, params: Optional[dict] = None) -> None:
    """Execute a non-SELECT SQL statement safely with parameter binding."""
    with engine.begin() as conn:                # start transaction
        conn.execute(text(sql), params or {})   # execute statement

def fetch_all(engine: Engine, sql: str, params: Optional[dict] = None) -> List[dict]:
    """Execute a SELECT query and return rows as list of dicts."""
    with engine.begin() as conn:                # start transaction
        result = conn.execute(text(sql), params or {})  # execute query
        return [dict(r._mapping) for r in result.fetchall()]  # convert to list of dicts

# -------------------------- Schema: pa_recordings --------------------------
DDL_PA_RECORDINGS = """
CREATE TABLE IF NOT EXISTS public.pa_recordings (
    id BIGSERIAL PRIMARY KEY,
    pa_recording_url TEXT UNIQUE NOT NULL,
    created TIMESTAMPTZ NOT NULL,              -- stores same timestamp as source table's created column
    pa_en_transcribe TEXT,
    call_type TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT now()       -- auto-generated timestamp when record was inserted
);
CREATE INDEX IF NOT EXISTS idx_pa_recordings_created ON public.pa_recordings (created);
"""

def init_db(engine: Optional[Engine] = None):
    """Ensure public.pa_recordings table exists (idempotent)."""
    engine_to_use = engine if engine is not None else get_engine()  # reuse engine when passed
    try:
        run_sql(engine_to_use, DDL_PA_RECORDINGS)   # create table and index if missing
        print(" pa_recordings table ensured/created successfully.")
    except SQLAlchemyError as e:
        print(f" Database initialization failed: {e}")
        raise

# ====================== OpenAI helper ======================
def _get_openai_client():
    """Return OpenAI client instance for modern SDK, or configure legacy module-level key."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    if _HAS_NEW_OPENAI:
        return OpenAI(api_key=OPENAI_API_KEY)      # return instantiated client
    else:
        openai.api_key = OPENAI_API_KEY            # configure legacy package
        return None

def transcribe_audio_file(local_file_path: str, prefer_translate: bool = True) -> str:
    """
    Transcribe or translate a local audio file to English using OpenAI audio endpoints.
    prefer_translate=True uses translations endpoint (better for non-English audio).
    """
    client = _get_openai_client()                  # obtain client or configure legacy
    try:
        if _HAS_NEW_OPENAI:                        # new SDK call shape
            with open(local_file_path, "rb") as f:
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
            return resp.strip() if isinstance(resp, str) else str(resp).strip()
        else:                                      # legacy SDK call shape
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
        print(f"[ASR] Transcription error for {local_file_path}: {e}")
        raise

# ====================== Helpers for recording_url processing ======================
def file_ext_from_url(url: str) -> str:
    """Return file extension from URL path or default to .mp3 when none present."""
    parsed = urlparse(url)                         # parse URL to extract path
    _, ext = os.path.splitext(parsed.path)        # get extension part
    return ext if ext else ".mp3"                 # fallback to .mp3

def is_allowed_extension_from_url(url: str) -> bool:
    """Return True if URL extension is among known audio extensions."""
    ext = file_ext_from_url(url).lower()          # lowercase extension
    return ext in ALLOWED_EXTS

def content_type_looks_like_audio(content_type: Optional[str]) -> bool:
    """Return True if Content-Type header appears audio-like or acceptable for ASR."""
    if not content_type:
        return False
    ct = content_type.lower()
    for sub in ALLOWED_CONTENT_TYPE_SUBSTRINGS:
        if sub in ct:
            return True
    return False

# ====================== Upsert into pa_recordings ======================
def upsert_pa_recording(engine: Engine, recording_url: str, created: datetime, transcript: str, call_type: str, status: str = "pending") -> None:
    """
    Insert or update a record in public.pa_recordings keyed by pa_recording_url.
    created stores the exact same timestamp as source table's created column.
    """
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
        "created": created,                       # store exact same timestamp as source
        "transcript": transcript,
        "call_type": call_type,
        "status": status
    })

# ====================== High-water-mark helper ======================
def max_pa_created(engine: Engine) -> Optional[str]:
    """Return ISO string of maximum created currently stored in pa_recordings (or None)."""
    rows = fetch_all(engine, "SELECT max(created) AS mx FROM public.pa_recordings")
    if not rows:
        return None
    mx = rows[0].get("mx")
    if mx is None:
        return None
    return mx.isoformat()                         # convert datetime to ISO format string

# ====================== Fetch source rows from outbound_inbound_calling_details ======================
def fetch_oicd_rows(engine: Engine, since_iso: Optional[str], limit: Optional[int] = None) -> List[dict]:
    """
    Fetch rows from outbound_inbound_calling_details with created > :since.
    If since_iso is None, compute high-water-mark from pa_recordings table or fallback to DEFAULT_SINCE_ISO.
    """
    if since_iso is None:                             # if user did not provide --since
        hw = max_pa_created(engine)                   # get maximum created processed so far from pa_recordings
        since_param = hw if hw else DEFAULT_SINCE_ISO # use hw if exists else fallback default
    else:
        since_param = since_iso                       # use explicit since provided

    base_sql = "SELECT * FROM public.outbound_inbound_calling_details oicd WHERE oicd.created > :since ORDER BY oicd.created ASC"
    if limit:
        base_sql += " LIMIT :limit"
        params = {"since": since_param, "limit": limit}
    else:
        params = {"since": since_param}

    return fetch_all(engine, base_sql, params)

# ====================== Main ingestion flow (HTTP-only, prefix support) ======================
def ingest_pa_http(since_iso: Optional[str] = None, limit: Optional[int] = None, prefer_translate: bool = True) -> None:
    """
    Core flow:
    - compute high-water-mark if since_iso is None
    - fetch candidate rows from outbound_inbound_calling_details
    - for each row: normalize recording value, rewrite to full URL if needed, validate, download, validate response, transcribe, upsert
    - always write status='pending' and created = source.created for processed rows (including failures)
    """
    if requests is None:
        raise RuntimeError("The 'requests' library is required. Install via `pip install requests`.")

    engine = get_engine()                             # create and reuse single SQLAlchemy engine
    init_db(engine)                                   # ensure pa_recordings exists (use same engine)

    since_param_for_log = since_iso if since_iso is not None else "(auto from pa_recordings.max(created) or default)"
    print(f" Determining source rows using since: {since_param_for_log}")  # log which since will be used

    rows = fetch_oicd_rows(engine, since_iso, limit=limit)  # fetch candidate rows
    print(f" Found {len(rows)} candidate rows.")

    processed = 0                                     # counters for summary
    succeeded = 0
    failed = 0
    skipped = 0

    for row in rows:                                  # iterate rows in ascending order of created
        processed += 1
        try:
            # tolerant extraction of fields from the source row
            raw_val = row.get("recording_url") or row.get("recordingUrl") or row.get("recording") or row.get("recordingLink")
            created = row.get("created")            # source 'created' timestamp (datetime from outbound_inbound_calling_details)
            call_type = row.get("call_type") or row.get("calltype") or row.get("direction") or row.get("call_type_io") or "unknown"
            source_id = row.get("id")               # source id for logs

            # Normalize early: convert to string and strip whitespace BEFORE any emptiness checks
            raw_val = "" if raw_val is None else str(raw_val).strip()
            print(f"\n[{processed}] Source id={source_id} Original DB value: {raw_val!r}")

            # If normalized value is empty after strip -> skip and record skipped
            if not raw_val:
                print(f"  Skipping id={source_id} — recording value is empty/whitespace")
                skipped += 1
                continue

            # Build full URL if DB stored a path-only value (no scheme) and prefix is provided
            parsed_initial = urlparse(raw_val)
            recording_url = raw_val
            if (not parsed_initial.scheme) and PA_RECORDING_HTTP_PREFIX:
                # safe-quote path portion and join with prefix
                quoted_path = quote(raw_val.lstrip("/"))
                recording_url = PA_RECORDING_HTTP_PREFIX + "/" + quoted_path
                print(f"  Rewrote DB path to full URL using PA_RECORDING_HTTP_PREFIX -> {recording_url}")

            parsed = urlparse(recording_url)        # parse final URL

            # If path ends with a slash (directory-like), skip — no filename to download
            if parsed.path.endswith("/"):
                print(f"  Skipping id={source_id} — URL looks like a directory (no file): {recording_url}")
                skipped += 1
                # still upsert pending record to advance high-water-mark (as per requirement)
                try:
                    upsert_pa_recording(engine, recording_url, created, transcript="", call_type=call_type, status="pending")
                    print(f"  Upserted pending record for id={source_id} (directory-like path).")
                except Exception as e:
                    print(f"  Failed to upsert pending record for id={source_id}: {e}")
                continue

            # Validate extension guard BEFORE trying to download (helps skip obvious non-audio)
            if not is_allowed_extension_from_url(recording_url):
                # extension is not among allowed list — still allow server to respond with proper content-type,
                # but prefer to skip to avoid known-bad files.
                print(f"  Warning id={source_id}: file extension not in allowed list: {file_ext_from_url(recording_url)}; attempting download but will validate content-type.")
                # do not increment skipped yet; attempt download and validate response headers below

            # Accept only http/https schemes
            if parsed.scheme not in ("http", "https"):
                print(f"  Skipping id={source_id} — unsupported URL scheme: {parsed.scheme}")
                skipped += 1
                continue

            # Download via HTTP streaming into a temporary file
            local_tmp_path = None
            try:
                print(f"  Downloading id={source_id}: {recording_url}")
                resp = requests.get(recording_url, stream=True, timeout=60)  # 60s timeout
                resp.raise_for_status()                                       # raise for HTTP errors
            except Exception as e:
                print(f"  HTTP download failed for id={source_id} url={recording_url}: {e}")
                failed += 1
                # Upsert pending record so high-water-mark can advance
                try:
                    upsert_pa_recording(engine, recording_url, created, transcript="", call_type=call_type, status="pending")
                    print(f"  Upserted pending row for id={source_id} after download failure.")
                except Exception as e2:
                    print(f"  Failed to upsert pending row for id={source_id} after download failure: {e2}")
                continue

            # Validate Content-Type header to ensure it's audio or acceptable binary
            content_type = resp.headers.get("content-type", "")
            content_length = resp.headers.get("content-length")
            if not content_type_looks_like_audio(content_type):
                # If content-type doesn't look audio, we still may accept application/octet-stream
                # but prefer to skip and upsert pending to avoid sending invalid files to ASR.
                print(f"  Downloaded resource Content-Type not audio-like for id={source_id}: {content_type!r}")
                # Consume body minimally to avoid leaving connection open then upsert pending
                try:
                    _ = resp.content  # read content to close connection
                except Exception:
                    pass
                failed += 1
                try:
                    upsert_pa_recording(engine, recording_url, created, transcript="", call_type=call_type, status="pending")
                    print(f"  Upserted pending row for id={source_id} due to non-audio Content-Type.")
                except Exception as e2:
                    print(f"  Failed to upsert pending row for id={source_id}: {e2}")
                continue

            # stream response to temp file and ensure size is reasonable
            try:
                ext = file_ext_from_url(recording_url)                 # extension for temp file naming
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="pa_audio_") as tmp:
                    written = 0
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            tmp.write(chunk)
                            written += len(chunk)
                    local_tmp_path = tmp.name                        # path of temp file
                # basic file size check to avoid tiny non-audio files
                if written < MIN_AUDIO_BYTES:
                    print(f"  Downloaded file too small for id={source_id} (bytes={written}), skipping ASR.")
                    # cleanup small file
                    try:
                        if local_tmp_path and os.path.exists(local_tmp_path):
                            os.remove(local_tmp_path)
                    except Exception:
                        pass
                    failed += 1
                    try:
                        upsert_pa_recording(engine, recording_url, created, transcript="", call_type=call_type, status="pending")
                        print(f"  Upserted pending row for id={source_id} due to tiny download.")
                    except Exception as e2:
                        print(f"  Failed to upsert pending row for id={source_id}: {e2}")
                    continue
            except Exception as e:
                print(f"  Failed while saving response to temp file for id={source_id}: {e}")
                failed += 1
                # cleanup if any partial file exists
                try:
                    if local_tmp_path and os.path.exists(local_tmp_path):
                        os.remove(local_tmp_path)
                except Exception:
                    pass
                try:
                    upsert_pa_recording(engine, recording_url, created, transcript="", call_type=call_type, status="pending")
                    print(f"  Upserted pending row for id={source_id} after temp-file write failure.")
                except Exception as e2:
                    print(f"  Failed to upsert pending row for id={source_id}: {e2}")
                continue

            # At this point we have a temporary file likely containing audio; call ASR
            try:
                print(f"  Transcribing id={source_id} file={local_tmp_path} (prefer_translate={prefer_translate})")
                transcript = transcribe_audio_file(local_tmp_path, prefer_translate=prefer_translate)
                print(f"  Transcription length for id={source_id}: {len(transcript)} characters")
                # Upsert transcript with status='pending' (per requirement) and created set to source.created
                upsert_pa_recording(engine, recording_url, created, transcript, call_type, status="pending")
                succeeded += 1
                print(f"  Upserted pending transcript for id={source_id} url={recording_url}")
            except Exception as e:
                print(f"  Transcription/upsert failed for id={source_id} url={recording_url}: {e}")
                failed += 1
                # ensure pending record is present so high-water-mark advances
                try:
                    upsert_pa_recording(engine, recording_url, created, transcript="", call_type=call_type, status="pending")
                    print(f"  Upserted pending row for id={source_id} after ASR failure.")
                except Exception as e2:
                    print(f"  Failed to upsert pending row for id={source_id} after ASR failure: {e2}")
            finally:
                # cleanup local temp file if created
                try:
                    if local_tmp_path and os.path.exists(local_tmp_path):
                        os.remove(local_tmp_path)
                except Exception as cleanup_err:
                    print(f"  Temp cleanup warning for id={source_id}: {cleanup_err}")

        except Exception as outer:
            # catch-all to avoid stopping the entire ingestion on single-row errors
            print(f"[{processed}] Unexpected error while processing source id={row.get('id')}: {outer}")
            failed += 1
            # attempt to write pending row with whatever URL info we have
            try:
                url_for_upsert = (row.get("recording_url") or "").strip()
                if not url_for_upsert and PA_RECORDING_HTTP_PREFIX:
                    url_for_upsert = PA_RECORDING_HTTP_PREFIX  # best-effort fallback
                upsert_pa_recording(engine, url_for_upsert, row.get("created") or datetime.now(timezone.utc), transcript="", call_type=row.get("call_type") or "unknown", status="pending")
                print(f"  Upserted fallback pending row for id={row.get('id')} after unexpected error.")
            except Exception as e2:
                print(f"  Failed fallback upsert after unexpected error for id={row.get('id')}: {e2}")
            continue

    # print ingest summary
    print("\n" + "=" * 60)
    print("PA INGESTION SUMMARY")
    print("=" * 60)
    print(f" Candidates fetched: {len(rows)}")
    print(f" Processed attempts: {processed}")
    print(f" Successful (upserted transcript): {succeeded}")
    print(f" Failed: {failed}")
    print(f" Skipped: {skipped}")
    print("=" * 60)

# ============================= CLI =============================
def main():
    """Parse CLI arguments and dispatch commands (init_db, ingest_pa)."""
    parser = argparse.ArgumentParser(description="PA Call Analyzer (HTTP-only) - ingest & transcribe recordings into pa_recordings")
    subs = parser.add_subparsers(dest="command", help="commands")

    subs.add_parser("init_db", help="Create/ensure public.pa_recordings table")

    p = subs.add_parser("ingest_pa", help="Fetch rows and transcribe recording_url via HTTP(S)")
    p.add_argument("--since", type=str, default=None, help="ISO timestamp boundary (default: auto high-water-mark from pa_recordings or fallback)")
    p.add_argument("--limit", type=int, default=None, help="Optional LIMIT for number of rows to process")
    p.add_argument("--no-translate", action="store_true", help="Use transcription (not translation) endpoint")
    p.add_argument("--prefer-translate", action="store_true", help="Prefer translation endpoint (overrides --no-translate)")

    args = parser.parse_args()

    if args.command == "init_db":
        init_db()                                  # ensure table exists (standalone)
        print(" Database init complete.")
        return

    if args.command == "ingest_pa":
        if not OPENAI_API_KEY:
            raise SystemExit("CRITICAL: OPENAI_API_KEY environment variable must be set.")
        if requests is None:
            raise SystemExit("CRITICAL: 'requests' library not installed. Install with `pip install requests`.")

        # choose prefer_translate precedence
        if args.prefer_translate:
            prefer_translate = True
        elif args.no_translate:
            prefer_translate = False
        else:
            prefer_translate = True

        # warn if prefix not set and DB likely contains path-only values
        if not PA_RECORDING_HTTP_PREFIX:
            print(" WARNING: PA_RECORDING_HTTP_PREFIX is not set. If DB stores only path/keys (not full URLs), downloads may fail.")

        # run ingestion; when args.since is None the script auto-computes high-water-mark
        ingest_pa_http(since_iso=args.since, limit=args.limit, prefer_translate=prefer_translate)
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
        raise SystemExit(1)