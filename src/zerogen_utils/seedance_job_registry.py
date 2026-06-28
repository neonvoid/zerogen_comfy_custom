"""Disk-backed job registry + per-job sidecar for non-blocking Seedance jobs.

The robustness backbone of the fire-and-forget submit/watch flow:

  - A CENTRAL registry (`<user>/zerogen_seedance_jobs/registry.json`) is the
    watcher's worklist — every submitted task_id + where its output should land.
    Survives ComfyUI restarts (the watcher resumes pending jobs from it on boot).
  - A PER-JOB sidecar JSON, written into the USER's output folder at SUBMIT time
    (status="submitted"), is the portable recovery receipt: it holds the task_id
    + full request even if the central registry is wiped or ComfyUI is reinstalled.
    BytePlus keeps the task 24h, so the sidecar's task_id is enough to recover the
    video by hand (Zerogen_SeedanceFetchTask) if everything else is lost.

No secrets are persisted — the API key is re-resolved from env/.env at poll time,
never written to disk.

All writes are atomic (temp file + os.replace) and guarded by a process lock so
the submit node and the background watcher can't corrupt the registry with an
interleaved read-modify-write.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime

# Statuses we still need to poll. Terminal statuses (succeeded/failed/cancelled/
# expired) are excluded — the watcher stops tracking them.
PENDING_STATUSES = ("submitted", "queued", "running")

_REGISTRY_DIRNAME = "zerogen_seedance_jobs"
_REGISTRY_FILENAME = "registry.json"

# Re-entrant: resolve_* helpers may be called while the lock is held elsewhere.
_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Location resolution
# ---------------------------------------------------------------------------

def registry_dir() -> str:
    """Stable, writable directory for the central registry. Prefers ComfyUI's
    user dir; falls back to <cwd>/user. Created on demand."""
    base = None
    try:
        import folder_paths  # type: ignore
        getter = getattr(folder_paths, "get_user_directory", None)
        if callable(getter):
            base = getter()
        else:
            base = os.path.dirname(folder_paths.get_output_directory())
    except Exception:
        base = None
    if not base:
        base = os.path.join(os.getcwd(), "user")
    d = os.path.join(base, _REGISTRY_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def registry_path() -> str:
    return os.path.join(registry_dir(), _REGISTRY_FILENAME)


# ---------------------------------------------------------------------------
# Atomic IO
# ---------------------------------------------------------------------------

def atomic_replace(tmp: str, dest: str, retries: int = 6) -> None:
    """os.replace with backoff. os.replace is atomic on the same filesystem on
    Windows + POSIX, BUT on Windows an AV scanner / indexer / preview watcher can
    briefly hold the destination open and raise PermissionError — so we retry a
    few times before giving up (review finding: both providers)."""
    for attempt in range(retries):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _atomic_write_json(path: str, obj) -> None:
    """Write JSON atomically: temp file in the same dir, then atomic_replace.

    A crash mid-write never leaves a half-written registry/sidecar.
    """
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Central registry CRUD (locked, read-modify-write)
# ---------------------------------------------------------------------------

def _load_unlocked() -> dict:
    data = _read_json(registry_path())
    if not isinstance(data, dict):
        return {}
    jobs = data.get("jobs")
    return jobs if isinstance(jobs, dict) else {}


def _save_unlocked(jobs: dict) -> None:
    _atomic_write_json(registry_path(), {"version": 1, "jobs": jobs})


def mutate_registry(mutator):
    """Atomic read-modify-write: hold the lock across load + mutate + save so a
    concurrent add/update can never clobber each other with a stale copy (review
    finding: separate load/save locks left a lost-update window). `mutator(jobs)`
    must be pure-synchronous (NO awaits, NO network) and may return a value, which
    this returns."""
    with _LOCK:
        jobs = _load_unlocked()
        result = mutator(jobs)
        _save_unlocked(jobs)
        return result


def load_registry() -> dict:
    """Return {task_id: entry}. Tolerates a missing or corrupt file (returns {})."""
    with _LOCK:
        return _load_unlocked()


def add_job(entry: dict) -> None:
    """Insert/replace a job entry keyed by its task_id."""
    tid = entry.get("task_id")
    if not tid:
        raise ValueError("add_job: entry has no task_id")

    def _mut(jobs):
        jobs[tid] = entry
    mutate_registry(_mut)


def update_job(task_id: str, **fields) -> dict | None:
    """Shallow-merge `fields` into an existing entry. Returns the merged entry
    (or None if the task_id is unknown — e.g. user cleared the registry)."""
    def _mut(jobs):
        entry = jobs.get(task_id)
        if entry is None:
            return None
        entry.update(fields)
        jobs[task_id] = entry
        return entry
    return mutate_registry(_mut)


def pending_jobs() -> list[dict]:
    """All entries still needing a poll (status in PENDING_STATUSES)."""
    return [e for e in load_registry().values() if e.get("status") in PENDING_STATUSES]


def has_pending() -> bool:
    return any(e.get("status") in PENDING_STATUSES for e in load_registry().values())


# ---------------------------------------------------------------------------
# Output path templating + preflight
# ---------------------------------------------------------------------------

_ILLEGAL_FN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_filename(name: str) -> str:
    """Strip characters illegal in Windows/POSIX filenames; collapse whitespace."""
    name = _ILLEGAL_FN.sub("_", str(name or "").strip())
    name = re.sub(r"\s+", "_", name)
    return name.strip("._") or "seedance"


def substitute_tokens(template: str, *, label: str, task_id: str, mode: str) -> str:
    """Replace {label} {task_id} {mode} {date} {time} {datetime} in a filename
    template. {date}/{time} use the LOCAL clock at resolution time."""
    now = datetime.now()
    repl = {
        "label": sanitize_filename(label) if label else "seedance",
        "task_id": sanitize_filename(task_id),
        "mode": sanitize_filename(mode) if mode else "gen",
        "date": now.strftime("%Y%m%d"),
        "time": now.strftime("%H%M%S"),
        "datetime": now.strftime("%Y%m%d-%H%M%S"),
    }
    out = template or "{label}_{task_id}"
    for k, v in repl.items():
        out = out.replace("{" + k + "}", v)
    # Any leftover unknown {tokens} → strip the braces, keep the word.
    out = re.sub(r"\{([^}]*)\}", r"\1", out)
    return sanitize_filename(out)


def preflight_output_dir(output_dir: str) -> str:
    """Expand, create, and verify the output dir is WRITABLE — at submit time, so
    a bad/unreachable path (e.g. a dropped Z: share) fails BEFORE we spend money,
    not 5 minutes later when the watcher tries to save. Returns the absolute path.

    Raises ValueError with an actionable message on any failure.
    """
    raw = (output_dir or "").strip()
    if not raw:
        raise ValueError("output_dir is empty — specify where the result should be saved.")
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(raw)))
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise ValueError(
            f"Cannot create output_dir {path!r}: {type(e).__name__}: {e}. "
            f"Check the drive/share is mounted and the path is valid."
        ) from e
    # Probe writability with a real temp file (permissions + share-liveness).
    try:
        fd, tmp = tempfile.mkstemp(dir=path, prefix=".write_test_")
        os.close(fd)
        os.remove(tmp)
    except OSError as e:
        raise ValueError(
            f"output_dir {path!r} is not writable: {type(e).__name__}: {e}. "
            f"If this is a network share (e.g. Z:), confirm it's connected."
        ) from e
    return path


def resolve_output_paths(
    *, output_dir: str, filename_template: str, label: str, task_id: str, mode: str
) -> dict:
    """Resolve the concrete destination paths from a (preflighted) dir + template.

    Returns {dir, base, video, last_frame, sidecar}. `output_dir` must already be
    absolute+validated (call preflight_output_dir first).
    """
    base = substitute_tokens(filename_template, label=label, task_id=task_id, mode=mode)
    stem = os.path.join(output_dir, base)
    return {
        "dir": output_dir,
        "base": base,
        "video": stem + ".mp4",
        "last_frame": stem + "_lastframe.png",
        "sidecar": stem + ".json",
    }


def write_sidecar(path: str, data: dict) -> None:
    """Atomic write of the per-job sidecar JSON (the portable recovery receipt)."""
    _atomic_write_json(path, data)
