"""Background watcher for non-blocking Seedance jobs.

A single long-lived asyncio task running on ComfyUI's OWN event loop
(`PromptServer.instance.loop`) — NOT a prompt execution. It never occupies the
queue / progress bar, so the UI stays free while it:

  1. polls every pending job in the disk registry,
  2. on success, downloads the video (+ optional last_frame) to the user's
     specified output folder,
  3. updates the per-job sidecar (submitted -> succeeded/failed),
  4. emits a websocket event (frontend toast) + a consolidated console heartbeat.

Robustness: the registry/sidecar on disk + the server-side task (24h) are the
source of truth. If this watcher crashes or ComfyUI restarts, nothing is lost —
the next submit re-spawns it and it resumes from the registry; worst case the
user recovers a job by task_id via Zerogen_SeedanceFetchTask.

No secrets on disk: the API key is re-resolved from env/.env each cycle.
"""

from __future__ import annotations

import asyncio
import threading
import time

import aiohttp

from . import seedance_job_registry as registry
from .api_keys import resolve_api_key
from .nv_byteplus_seedance_gen import _API_BASE, _STATUS_PATH

# Non-terminal vs terminal task vocabulary (mirrors the gen node's poll loop).
_NONTERMINAL = ("queued", "running")
_TERMINAL_OK = "succeeded"
_TERMINAL_BAD = ("failed", "cancelled", "expired")

_IDLE_SLEEP_S = 5.0
_MAX_IDLE_CYCLES = 6          # ~30s of no jobs -> watcher stops (re-spawned on next submit)
_MAX_CONCURRENT_POLLS = 8
_DOWNLOAD_RETRIES = 3         # in-cycle retries within one download attempt
_MAX_DOWNLOAD_ATTEMPTS = 5   # across cycles, before a succeeded job is given up as download_failed
_JOB_MAX_AGE_S = 24 * 3600   # BytePlus purges task + result URL after 24h — stop polling past this

# Singleton state
_watcher_task: asyncio.Task | None = None
_watcher_lock = threading.Lock()
_poll_interval_s = 10.0
_verbosity = "normal"
_VERBOSITY_RANK = {"quiet": 0, "normal": 1, "debug": 2}


# ---------------------------------------------------------------------------
# Logging + notification
# ---------------------------------------------------------------------------

def _vlog(level: str, msg: str) -> None:
    """Console log gated by the active verbosity (quiet < normal < debug)."""
    if _VERBOSITY_RANK.get(_verbosity, 1) >= _VERBOSITY_RANK.get(level, 1):
        print(f"[SeedanceWatcher] {msg}")


def _notify(event: str, payload: dict) -> None:
    """Best-effort websocket push to the frontend (toast). Never raises."""
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(event, payload)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def ensure_watcher_running(poll_interval_s: float = 10.0, verbosity: str = "normal") -> bool:
    """Idempotently ensure the watcher task is alive on ComfyUI's event loop.

    Safe to call from any submit. Updates the live poll interval / verbosity for
    the running loop. Returns True if the watcher is (now) scheduled, False if the
    loop isn't available yet (headless import before the server starts).
    """
    global _watcher_task, _poll_interval_s, _verbosity
    _poll_interval_s = float(poll_interval_s)
    _verbosity = verbosity if verbosity in _VERBOSITY_RANK else "normal"
    try:
        from server import PromptServer
        loop = PromptServer.instance.loop
    except Exception:
        loop = None
    if loop is None:
        return False

    def _start() -> None:
        global _watcher_task
        if _watcher_task is None or _watcher_task.done():
            _watcher_task = loop.create_task(_watcher_loop())
            _watcher_task.add_done_callback(_on_watcher_done)

    with _watcher_lock:
        if _watcher_task is not None and not _watcher_task.done():
            return True
        try:
            loop.call_soon_threadsafe(_start)
        except RuntimeError:
            return False
        return True


def _on_watcher_done(task: "asyncio.Task") -> None:
    """Done-callback: log a crash + SELF-HEAL. If jobs are still pending when the
    watcher exits — whether from an unhandled exception or an idle-stop that raced
    with a fresh submit — respawn it so a paid job is never orphaned (review
    finding: idle-stop/respawn race + watcher-death-on-exception)."""
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    except Exception:
        exc = None
    if exc is not None:
        print(f"[SeedanceWatcher] task exited with exception: {type(exc).__name__}: {exc}")
    try:
        if registry.has_pending():
            _vlog("normal", "pending jobs remain after exit — respawning watcher")
            ensure_watcher_running(_poll_interval_s, _verbosity)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

async def _watcher_loop() -> None:
    _vlog("normal", "started")
    idle_cycles = 0
    try:
        while True:
            jobs = registry.pending_jobs()
            if not jobs:
                idle_cycles += 1
                if idle_cycles >= _MAX_IDLE_CYCLES:
                    # Final recheck — a submit may have landed a job since the read
                    # above. Belt-and-suspenders with the done-callback respawn.
                    if registry.has_pending():
                        idle_cycles = 0
                        continue
                    _vlog("normal", "idle — 0 jobs in flight; stopping (re-spawns on next submit)")
                    return
                await asyncio.sleep(_IDLE_SLEEP_S)
                continue
            idle_cycles = 0
            await _poll_cycle(jobs)
            _heartbeat()
            await asyncio.sleep(_poll_interval_s)
    except asyncio.CancelledError:
        _vlog("normal", "cancelled")
        raise
    except Exception as e:  # noqa: BLE001 — never let the watcher die silently
        print(f"[SeedanceWatcher] loop error: {type(e).__name__}: {e} — will restart on next submit")


async def _poll_cycle(jobs: list[dict]) -> None:
    """Poll all pending jobs concurrently within one short-lived session."""
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=120)
    connector = aiohttp.TCPConnector(force_close=True, limit=_MAX_CONCURRENT_POLLS)
    sem = asyncio.Semaphore(_MAX_CONCURRENT_POLLS)
    session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    try:
        async def _guarded(entry):
            async with sem:
                try:
                    await _process_job(session, entry)
                except Exception as e:  # noqa: BLE001 — one bad job must not stall the rest
                    _vlog("debug", f"{entry.get('label')} poll error (will retry): {type(e).__name__}: {e}")
        await asyncio.gather(*[_guarded(e) for e in jobs])
    finally:
        await session.close()
        await asyncio.sleep(0.05)  # Windows ProactorEventLoop SSL cleanup tick


async def _process_job(session: aiohttp.ClientSession, entry: dict) -> None:
    task_id = entry["task_id"]
    label = entry.get("label") or task_id
    # 24h circuit breaker: BytePlus purges the task + result URL after 24h, so
    # polling past that is pointless. Stop tracking (review finding: don't poll forever).
    age = time.time() - float(entry.get("submitted_at_epoch") or time.time())
    if age > _JOB_MAX_AGE_S:
        registry.update_job(task_id, status="failed", error="expired (>24h — BytePlus retention elapsed)")
        _vlog("quiet", f"✗ {label} expired (>24h, task purged server-side) — task_id={task_id}")
        _notify("zerogen.seedance.failed", {"task_id": task_id, "label": label,
                                            "status": "expired", "error": "24h retention elapsed"})
        return
    key = resolve_api_key("", provider="volcengine")
    headers = {"Authorization": f"Bearer {key}"}
    url = f"{_API_BASE}{_STATUS_PATH}/{task_id}"
    async with session.get(url, headers=headers) as resp:
        body = await resp.text()
        if resp.status != 200:
            # Transient (5xx/408/429) or a real error — either way, leave the job
            # pending and retry next cycle. The deadline is the user's problem to
            # cancel; we don't forfeit a paid job on a poll blip.
            _vlog("debug", f"{label}: status HTTP {resp.status} (retry next cycle)")
            return
        import json as _json
        data = _json.loads(body)

    status = data.get("status")
    if status in _NONTERMINAL:
        # Record running-state transition (submitted -> queued/running) once.
        if entry.get("status") != status:
            registry.update_job(task_id, status=status)
            _notify("zerogen.seedance.update", {"task_id": task_id, "label": label, "status": status})
        return

    if status == _TERMINAL_OK:
        await _collect_success(session, entry, data)
        return

    if status in _TERMINAL_BAD:
        _finalize_failure(entry, data)
        return

    # Unknown status — stop tracking, surface loudly (don't poll forever).
    _vlog("normal", f"{label}: UNKNOWN status {status!r} — dropping from tracking. task_id={task_id}")
    registry.update_job(task_id, status="failed", error=f"unknown status {status!r}")


# ---------------------------------------------------------------------------
# Terminal handling
# ---------------------------------------------------------------------------

async def _collect_success(session: aiohttp.ClientSession, entry: dict, data: dict) -> None:
    task_id = entry["task_id"]
    label = entry.get("label") or task_id
    content = data.get("content") or {}
    video_url = content.get("video_url")
    last_frame_url = content.get("last_frame_url")
    paths = entry.get("paths") or {}

    if not video_url:
        _vlog("normal", f"✗ {label}: succeeded but no video_url — recover via FetchTask. task_id={task_id}")
        registry.update_job(task_id, status="failed", error="succeeded but content.video_url missing")
        return

    saved_video = None
    try:
        saved_video = await _download_to_file(session, video_url, paths.get("video"))
    except Exception as e:  # noqa: BLE001
        # Save failed (e.g. Z: dropped). Retry across cycles while the 24h URL is
        # alive, but bounded — don't retry forever (review finding: circuit breaker).
        attempts = int(entry.get("download_attempts", 0)) + 1
        registry.update_job(task_id, download_attempts=attempts,
                            last_download_error=f"{type(e).__name__}: {e}")
        if attempts >= _MAX_DOWNLOAD_ATTEMPTS:
            registry.update_job(task_id, status="failed",
                                error=f"download failed {attempts}x: {e}")
            _vlog("quiet", f"✗ {label}: download failed {attempts}x — giving up "
                           f"(result still on server ~24h; recover via FetchTask). task_id={task_id}")
            _notify("zerogen.seedance.failed", {"task_id": task_id, "label": label,
                                                "status": "download_failed", "error": str(e)})
        else:
            _vlog("normal", f"⚠ {label}: download/save failed (attempt {attempts}/{_MAX_DOWNLOAD_ATTEMPTS}: "
                            f"{type(e).__name__}: {e}) — will retry next cycle")
        return

    saved_lf = None
    if entry.get("save_last_frame") and last_frame_url and paths.get("last_frame"):
        try:
            saved_lf = await _download_to_file(session, last_frame_url, paths["last_frame"])
        except Exception as e:  # noqa: BLE001 — last_frame is optional, don't fail the job
            _vlog("debug", f"{label}: last_frame save failed: {type(e).__name__}: {e}")

    elapsed = round(time.time() - float(entry.get("submitted_at_epoch") or time.time()), 1)
    usage = data.get("usage") or {}
    registry.update_job(
        task_id, status="succeeded", saved_video=saved_video, saved_last_frame=saved_lf,
        completed_at=_now_iso(), elapsed_s=elapsed,
    )
    _update_sidecar(entry, status="succeeded", response={
        "status": "succeeded",
        # Persist the URL PATH only (drop the query) — the presigned X-Tos-Signature
        # is a 24h capability token; no need to write secrets to disk. Recovery is by
        # task_id via FetchTask, which mints a fresh signed URL.
        "video_url": _strip_query(video_url),
        "last_frame_url": _strip_query(last_frame_url),
        "saved_video": saved_video,
        "saved_last_frame": saved_lf,
        "elapsed_s": elapsed,
        "total_tokens": usage.get("total_tokens"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
    })
    _vlog("quiet", f"✓ {label} succeeded in {elapsed}s -> {saved_video}")
    _notify("zerogen.seedance.done", {
        "task_id": task_id, "label": label, "saved_video": saved_video,
        "saved_last_frame": saved_lf, "elapsed_s": elapsed,
    })


def _finalize_failure(entry: dict, data: dict) -> None:
    task_id = entry["task_id"]
    label = entry.get("label") or task_id
    status = data.get("status")
    err = data.get("error") or {}
    code = err.get("code")
    msg = err.get("message")
    hint = ""
    if code and ("SensitiveContent" in str(code) or "PolicyViolation" in str(code)):
        hint = " (OUTPUT content gate — face/IP-driven, not asset-bypassable)"
    registry.update_job(task_id, status=status, error=f"{code}: {msg}{hint}")
    _update_sidecar(entry, status=status, response={
        "status": status, "error_code": code, "error_message": msg,
    })
    _vlog("quiet", f"✗ {label} {status} — {code}: {msg}{hint} — task_id={task_id} (recover/retry via FetchTask)")
    _notify("zerogen.seedance.failed", {
        "task_id": task_id, "label": label, "status": status,
        "error": f"{code}: {msg}",
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _download_to_file(session: aiohttp.ClientSession, url: str, path: str | None) -> str:
    """Download `url` to `path` atomically (temp + replace). Retries on transient
    network errors. Returns the saved path. Raises on final failure."""
    if not path:
        raise ValueError("no destination path resolved for this job")
    import os
    import tempfile
    last_exc = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status} downloading result")
                data = await resp.read()
            d = os.path.dirname(path) or "."
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_dl_")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                registry.atomic_replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            return path
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < _DOWNLOAD_RETRIES:
                await asyncio.sleep(min(8.0, 2.0 * attempt))
    raise RuntimeError(f"download failed after {_DOWNLOAD_RETRIES} attempts: {last_exc}")


def _update_sidecar(entry: dict, *, status: str, response: dict) -> None:
    """Merge terminal results into the per-job sidecar in the user's folder."""
    sidecar_path = (entry.get("paths") or {}).get("sidecar")
    if not sidecar_path:
        return
    existing = registry._read_json(sidecar_path) or {}
    existing["status"] = status
    existing["response"] = response
    existing["task_id"] = entry["task_id"]
    try:
        registry.write_sidecar(sidecar_path, existing)
    except Exception as e:  # noqa: BLE001 — sidecar is a convenience, registry is authoritative
        _vlog("debug", f"sidecar update failed for {entry.get('label')}: {type(e).__name__}: {e}")


def _heartbeat() -> None:
    """One consolidated status line per cycle for all in-flight jobs."""
    if _VERBOSITY_RANK.get(_verbosity, 1) < _VERBOSITY_RANK["normal"]:
        return
    jobs = registry.pending_jobs()
    if not jobs:
        return
    now = time.time()
    parts = []
    for e in jobs:
        el = int(now - float(e.get("submitted_at_epoch") or now))
        parts.append(f"{e.get('label') or e['task_id']}: {e.get('status','submitted')} {el}s")
    _vlog("normal", f"{len(jobs)} in flight | " + " | ".join(parts))


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def _strip_query(url: str | None) -> str | None:
    """Drop the query string (presigned signature material) from a URL before
    persisting it to disk. Keeps the object path for 'which file' visibility."""
    if not url:
        return url
    return url.split("?", 1)[0]
