"""Persistent local cache for BytePlus (native) asset library registrations.

Sibling of `nv_moyu_asset_cache.py` with its OWN cache file — the Moyu and
BytePlus node groups must never share fate (operator constraint 2026-06-11),
and the two libraries have disjoint asset-id namespaces anyway.

Same architecture rationale as the Moyu cache: separate asset REGISTRATION
(one-time, network) from asset USE (repeated, free). Content SHA256 is the
identity; the cache returns the `asset://` URL instantly on re-runs.

One improvement over the Moyu sibling is enabled by the API (implemented in
the register node, not here): BytePlus `GetAsset` is authoritative and cheap
(100 QPS), so cached entries can be liveness-verified before use and
self-healed via `invalidate()` — fixing the known Moyu-cache flaw of serving
dead asset IDs forever.

Per-AK partitioning: entries are namespaced by a fingerprint of the
ARK_ACCESS_KEY. Key rotation = fresh namespace = re-register (the remote-adopt
tier in the register node makes that cheap — it finds the existing asset by
deterministic name instead of re-uploading).

Cache file format (JSON):
```
{
  "version": 1,
  "entries": {
    "<ak_fingerprint>": {
      "<sha256_hash>": {
        "asset_id": "asset-...",
        "asset_url": "asset://asset-...",
        "asset_type": "Image" | "Video",
        "tag": "optional human-friendly label",
        "registered_at": <unix_ts>,
        "group_id": "group-...",
        "group_name": "<group name>",
        "project_name": "default",
        "source_url_tail": "<last 40 chars of staging URL>",
        "verified_at": <unix_ts | null>
      }
    }
  }
}
```

Atomic write: write to `<cache>.tmp` → flush → fsync → os.replace. Crash-safe
and torn-read-safe for parallel agents (same guarantees as the Moyu cache,
see its docstring for the multi-AI-reviewed reasoning).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Cache file location
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_PATH = Path.home() / ".nv_comfy_utils" / "byteplus_assets.json"

_SCHEMA_VERSION = 1


def get_cache_path() -> Path:
    """Resolve cache file path. Honors env var NV_BYTEPLUS_ASSET_CACHE_PATH for override."""
    override = os.environ.get("NV_BYTEPLUS_ASSET_CACHE_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_CACHE_PATH


def _key_fingerprint(access_key: str) -> str:
    """Stable fingerprint of the ARK access key for cache partitioning.

    First 8 + last 4 chars hashed (never stores the full key). Empty key falls
    back to a sentinel namespace so debug/dry-run flows still work.
    """
    if not access_key or not access_key.strip():
        return "no-key"
    k = access_key.strip()
    sample = k[:8] + k[-4:] if len(k) >= 12 else k
    return hashlib.sha1(sample.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def _load_raw() -> dict[str, Any]:
    """Load the cache file; empty bootstrap dict on missing/corrupt (graceful degradation)."""
    path = get_cache_path()
    if not path.is_file():
        return {"version": _SCHEMA_VERSION, "entries": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[NV_ByteplusAssetCache] WARN: failed to read cache {path} ({e}); treating as empty.")
        return {"version": _SCHEMA_VERSION, "entries": {}}
    if not isinstance(data, dict) or "entries" not in data:
        print(f"[NV_ByteplusAssetCache] WARN: cache schema unrecognized at {path}; treating as empty.")
        return {"version": _SCHEMA_VERSION, "entries": {}}
    if data.get("version") != _SCHEMA_VERSION:
        print(f"[NV_ByteplusAssetCache] WARN: cache version {data.get('version')!r} != {_SCHEMA_VERSION}; treating as empty.")
        return {"version": _SCHEMA_VERSION, "entries": {}}
    return data


def _save_raw(data: dict[str, Any]) -> None:
    """Atomic + durable write: tmp → flush → fsync → os.replace (see Moyu sibling docstring)."""
    path = get_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(serialized)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Public API — keyed by content SHA256
# ---------------------------------------------------------------------------


def lookup(access_key: str, content_hash: str) -> dict[str, Any] | None:
    """Return the cached entry for (access_key, content_hash) or None."""
    data = _load_raw()
    fp = _key_fingerprint(access_key)
    ns = data.get("entries", {}).get(fp, {})
    entry = ns.get(content_hash)
    if isinstance(entry, dict):
        return entry
    return None


def register(
    access_key: str,
    content_hash: str,
    *,
    asset_id: str,
    asset_url: str,
    asset_type: str,
    tag: str = "",
    group_id: str = "",
    group_name: str = "",
    project_name: str = "default",
    source_url_tail: str = "",
) -> dict[str, Any]:
    """Persist a registration. Idempotent — same (key, hash) replaces prior entry."""
    data = _load_raw()
    fp = _key_fingerprint(access_key)
    entries = data.setdefault("entries", {})
    ns = entries.setdefault(fp, {})
    now = int(time.time())
    entry = {
        "asset_id": asset_id,
        "asset_url": asset_url,
        "asset_type": asset_type,
        "tag": tag,
        "registered_at": now,
        "verified_at": now,
        "group_id": group_id,
        "group_name": group_name,
        "project_name": project_name,
        "source_url_tail": source_url_tail[-40:] if source_url_tail else "",
    }
    ns[content_hash] = entry
    _save_raw(data)
    return entry


def mark_verified(access_key: str, content_hash: str) -> None:
    """Update `verified_at` after a successful GetAsset liveness check. No-op if missing."""
    data = _load_raw()
    fp = _key_fingerprint(access_key)
    entry = data.get("entries", {}).get(fp, {}).get(content_hash)
    if not isinstance(entry, dict):
        return
    entry["verified_at"] = int(time.time())
    _save_raw(data)


def invalidate(access_key: str, content_hash: str) -> bool:
    """Remove a cache entry (self-heal path: GetAsset said Failed / NotFound).

    Returns True if an entry was removed.
    """
    data = _load_raw()
    fp = _key_fingerprint(access_key)
    ns = data.get("entries", {}).get(fp, {})
    if content_hash in ns:
        del ns[content_hash]
        _save_raw(data)
        return True
    return False


def list_entries(access_key: str) -> dict[str, dict[str, Any]]:
    """All entries for the given access key as `{content_hash: entry_dict}`."""
    data = _load_raw()
    fp = _key_fingerprint(access_key)
    ns = data.get("entries", {}).get(fp, {})
    if isinstance(ns, dict):
        return dict(ns)
    return {}


# ---------------------------------------------------------------------------
# Content hashing — same semantics as the Moyu sibling (duplicated, not
# imported, so the two node groups stay fully decoupled). Same content
# produces the same hash in BOTH caches by construction.
# ---------------------------------------------------------------------------


def hash_image_tensor(tensor) -> str:  # noqa: ANN001 — torch tensor
    """SHA256 of an IMAGE tensor's raw bytes. Stable across processes.

    `.contiguous()` because batch-sliced / transposed tensors raise
    "ndarray is not C-contiguous" on `.tobytes()` (R0 review).
    """
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def hash_video_input(video) -> str:  # noqa: ANN001 — Input.Video duck-typed
    """SHA256 of a VIDEO signature (first frame + last frame + frame count)."""
    try:
        components = video.get_components()
        frames = components.images
        n = int(frames.shape[0])
        if n == 0:
            return f"nohash_emptyvideo_{int(time.time() * 1e9)}"
        first = frames[0].detach().contiguous().cpu().numpy().tobytes()
        last = frames[-1].detach().contiguous().cpu().numpy().tobytes()
        sig = first + last + str(n).encode("utf-8")
        return hashlib.sha256(sig).hexdigest()
    except Exception:
        return f"nohash_videoerr_{int(time.time() * 1e9)}"
