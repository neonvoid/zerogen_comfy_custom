"""BytePlus ModelArk Assets Action API client — V4 signing + asset helpers.

Runtime-validated 2026-06-11: full console-free loop proven —
local file -> B2 presigned URL -> CreateAsset -> GetAsset Active (~10s image)
-> `asset://<id>` generation succeeded.

This module is the NATIVE (BytePlus international) sibling of
`nv_seedance_moyu_api_helpers.py`. It is deliberately a SEPARATE module with
zero imports from the Moyu helper stack — the two node groups must never
share fate (design constraint 2026-06-11: existing Moyu workflows must not
change at all).

Two credentials, two planes:
- Generation plane: Bearer ARK_API_KEY at ark.ap-southeast.bytepluses.com/api/v3
  (NOT this module's job — the native gen nodes own that).
- Asset Action plane (THIS module): AK/SK pair (ARK_ACCESS_KEY/ARK_SECRET_KEY),
  Volcengine V4 request signing, ServiceName `ark`, Version `2024-01-01`,
  region `ap-southeast-1`. Actions: CreateAssetGroup / CreateAsset / GetAsset /
  ListAssets / ListAssetGroups / DeleteAsset / ...

HARD CONSTRAINT (docs explicit + empirically confirmed): CreateAsset is
URL-pull ONLY. "For image/video/audio assets, only URL upload is supported.
Base64 is not supported." A `data:` URI returns InvalidParameter.URL. The
caller must stage bytes at a fetchable HTTP(S) URL (B2 presigned) first; once
the asset reaches Active the file lives in ByteDance TOS and the staging
object can be deleted immediately.

Library semantics:
- Virtual Portrait library = GroupType `AIGC` — NO liveness, accepts real
  faces, NO same-person consistency check between assets in a group.
- Real-human library = GroupType `LivenessFace` — H5 liveness QR + per-asset
  face-consistency enforcement. NOT used by these helpers (different flow).
- Assets are project-scoped (ProjectName, default `default`) and must live in
  the same project as the generation endpoint.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import hmac
import json
import os
from typing import Any

import aiohttp

from .api_keys import _load_dotenv_once

# ---------------------------------------------------------------------------
# Constants — Action API plane
# ---------------------------------------------------------------------------

ARK_REGION = "ap-southeast-1"
ARK_SERVICE = "ark"
ARK_ACTION_VERSION = "2024-01-01"
# Documented host. `open.ap-southeast-1.byteplusapi.com` is an alias that also
# works (validated 2026-06-11); we use the documented one and keep the alias
# as fallback for DNS hiccups.
ARK_ACTION_HOSTS = ("ark.ap-southeast-1.byteplusapi.com", "open.ap-southeast-1.byteplusapi.com")

GROUP_TYPE_AIGC = "AIGC"

DEFAULT_PROJECT_NAME = "default"
DEFAULT_ASSET_GROUP_NAME = "nv_refs"

# Deterministic per-content asset name — the remote-adopt tier lists by this
# name, so a lost local cache never causes a duplicate upload.
ASSET_NAME_PREFIX = "nv-ref-"


def deterministic_asset_name(content_hash: str) -> str:
    """`nv-ref-{hash[:24]}` — stable identity for remote-adopt lookups.

    24 hex chars (96 bits) per R0 review: 12 was a theoretical adopt-the-wrong-
    asset collision risk; 31 total chars stays well under the 64-char Name cap.
    """
    return f"{ASSET_NAME_PREFIX}{content_hash[:24]}"


# ---------------------------------------------------------------------------
# Credential resolution — AK/SK pair (NOT the Bearer key)
# ---------------------------------------------------------------------------


def resolve_ark_ak_sk(access_key: str = "", secret_key: str = "") -> tuple[str, str]:
    """Resolve the ARK AK/SK pair: explicit input > env > Comfy_Utils/.env.

    Raises RuntimeError with remediation guidance if either half is missing.
    The IAM user needs ArkFullAccess on the target project (validated:
    `zs-asset-1`).
    """
    ak = (access_key or "").strip()
    sk = (secret_key or "").strip()
    if not ak or not sk:
        _load_dotenv_once()
        if not ak:
            ak = (os.environ.get("ARK_ACCESS_KEY") or "").strip()
        if not sk:
            sk = (os.environ.get("ARK_SECRET_KEY") or "").strip()
    if not ak or not sk:
        raise RuntimeError(
            "No BytePlus AK/SK pair for the asset Action API. Either:\n"
            "  - Set ARK_ACCESS_KEY and ARK_SECRET_KEY environment variables, or\n"
            "  - Add them to Comfy_Utils/.env, or\n"
            "  - Paste them into the ark_access_key / ark_secret_key node inputs.\n"
            "Note: this is the IAM Access Key pair (console > IAM > Access Keys), "
            "NOT the Bearer ARK_API_KEY used by the generation endpoint."
        )
    return ak, sk


# ---------------------------------------------------------------------------
# Volcengine/BytePlus V4 request signing (SigV4 variant, terminator `request`)
#
# Pure function so unit tests can pin a known-good signature vector with a
# fixed timestamp. Runtime-validated against the live API 2026-06-11.
# ---------------------------------------------------------------------------


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def build_v4_signed_headers(
    *,
    action: str,
    payload: bytes,
    access_key: str,
    secret_key: str,
    host: str,
    x_date: str,
    region: str = ARK_REGION,
    service: str = ARK_SERVICE,
    version: str = ARK_ACTION_VERSION,
) -> dict[str, str]:
    """Build the signed header set for one POST Action call.

    `x_date` is `YYYYMMDDTHHMMSSZ` (UTC). Canonical request shape:
    POST / Action=<action>&Version=<version> over signed headers
    content-type;host;x-content-sha256;x-date, scope
    `{yyyymmdd}/{region}/{service}/request`.
    """
    short_date = x_date[:8]
    content_sha = hashlib.sha256(payload).hexdigest()
    query = f"Action={action}&Version={version}"

    canonical_headers = (
        f"content-type:application/json\n"
        f"host:{host}\n"
        f"x-content-sha256:{content_sha}\n"
        f"x-date:{x_date}\n"
    )
    signed_headers = "content-type;host;x-content-sha256;x-date"
    canonical_request = "\n".join(["POST", "/", query, canonical_headers, signed_headers, content_sha])
    scope = f"{short_date}/{region}/{service}/request"
    string_to_sign = "\n".join(["HMAC-SHA256", x_date, scope, hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()])
    k_signing = _hmac_sha256(_hmac_sha256(_hmac_sha256(_hmac_sha256(secret_key.encode("utf-8"), short_date), region), service), "request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    return {
        "Content-Type": "application/json",
        "Host": host,
        "X-Date": x_date,
        "X-Content-Sha256": content_sha,
        "Authorization": (
            f"HMAC-SHA256 Credential={access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
    }


_MUTATING_ACTION_PREFIXES = ("Create", "Delete", "Update")


def _is_mutating_action(action: str) -> bool:
    """Actions that change remote state — never replayed on ambiguous failures."""
    return action.startswith(_MUTATING_ACTION_PREFIXES)


class ArkActionError(RuntimeError):
    """Structured error from the Action API envelope (ResponseMetadata.Error)."""

    def __init__(self, action: str, code: str, message: str, http_status: int):
        self.action = action
        self.code = code
        self.message = message
        self.http_status = http_status
        super().__init__(f"{action} failed (HTTP {http_status}) {code}: {message}")


async def action_call(
    session: aiohttp.ClientSession,
    action: str,
    body: dict[str, Any],
    access_key: str,
    secret_key: str,
) -> dict[str, Any]:
    """POST one signed Action call; return the `Result` dict (or {} if absent).

    Raises ArkActionError on an error envelope, aiohttp errors on transport
    failure. Tries the documented host first, the alias on connection errors.
    """
    payload = json.dumps(body).encode("utf-8")
    last_exc: Exception | None = None
    for host in ARK_ACTION_HOSTS:
        x_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        headers = build_v4_signed_headers(
            action=action, payload=payload, access_key=access_key, secret_key=secret_key, host=host, x_date=x_date,
        )
        try:
            async with session.post(f"https://{host}/?Action={action}&Version={ARK_ACTION_VERSION}", data=payload, headers=headers) as r:
                text = await r.text()
                try:
                    data = json.loads(text or "{}")
                except json.JSONDecodeError:
                    raise ArkActionError(action, "UnparseableResponse", text[:300], r.status) from None
                meta = data.get("ResponseMetadata") or {}
                err = meta.get("Error")
                if err:
                    raise ArkActionError(action, err.get("Code", "Unknown"), err.get("Message", ""), r.status)
                if r.status >= 400:
                    raise ArkActionError(action, "HTTPError", text[:300], r.status)
                result = data.get("Result")
                return result if isinstance(result, dict) else {}
        except (aiohttp.ClientConnectionError, aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
            # DNS / connect failure / hang on this host → try the alias host
            # (R0 review convergent: a blackholed primary that times out should
            # fall through, not blow up the call). API-envelope errors propagate.
            #
            # R1 (Codex): MUTATING actions must NOT be replayed after an
            # ambiguous failure — a read-timeout can happen AFTER the server
            # accepted the request, and a second CreateAsset would duplicate
            # the asset. Only ClientConnectorError (connect-phase, request
            # provably never sent) may fall through for those.
            if _is_mutating_action(action) and not isinstance(e, aiohttp.ClientConnectorError):
                raise
            print(f"[ByteplusAssetAPI] {action}: host {host} unreachable ({e.__class__.__name__}); trying fallback host.")
            last_exc = e
            continue
    raise last_exc if last_exc else RuntimeError(f"{action}: no hosts attempted")


async def _list_all_pages(
    session: aiohttp.ClientSession,
    action: str,
    base_body: dict[str, Any],
    access_key: str,
    secret_key: str,
    *,
    page_size: int = 50,
    max_pages: int = 20,
):
    """Yield Items across pages until a short page (R0 review: the server-side
    Name filter is FUZZY, so an exact match can sit past page 1)."""
    for page in range(1, max_pages + 1):
        body = dict(base_body)
        body["PageNumber"] = page
        body["PageSize"] = page_size
        result = await action_call(session, action, body, access_key, secret_key)
        items = result.get("Items") or []
        for item in items:
            yield item
        if len(items) < page_size:
            return
    print(f"[ByteplusAssetAPI] WARN: {action} pagination stopped at {max_pages} pages; exact match may be missed.")


# ---------------------------------------------------------------------------
# Asset helpers built on action_call
# ---------------------------------------------------------------------------


def looks_like_group_id(ref: str) -> bool:
    """`group-...` references are used as-is; anything else is a group NAME."""
    return ref.strip().lower().startswith("group-")


async def ensure_asset_group(
    session: aiohttp.ClientSession,
    access_key: str,
    secret_key: str,
    group_ref: str,
    *,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> str:
    """Resolve a group reference (id or name) to a GroupId; create-if-missing by name.

    - `group-...` → returned verbatim (no extra round-trip; CreateAsset will
      fail loudly if it doesn't exist).
    - name → exact match against ListAssetGroups (GroupType AIGC, fuzzy server
      search then exact filter client-side) → CreateAssetGroup(AIGC) if absent.

    Note: the account must have signed the one-time asset authorization letter
    in the console before CreateAssetGroup works (already done on the studio
    account, 2026-06-11).
    """
    ref = group_ref.strip()
    if not ref:
        ref = DEFAULT_ASSET_GROUP_NAME
    if looks_like_group_id(ref):
        return ref

    async def _find_exact() -> str | None:
        async for item in _list_all_pages(
            session, "ListAssetGroups",
            {"Filter": {"Name": ref, "GroupType": GROUP_TYPE_AIGC}},
            access_key, secret_key,
        ):
            if item.get("Name") == ref and item.get("ProjectName", DEFAULT_PROJECT_NAME) == project_name:
                return str(item["Id"])
        return None

    found = await _find_exact()
    if found is not None:
        return found

    print(f"[ByteplusAssetAPI] Asset group {ref!r} not found; creating (GroupType={GROUP_TYPE_AIGC}, project={project_name}).")
    try:
        created = await action_call(
            session, "CreateAssetGroup",
            {"Name": ref, "GroupType": GROUP_TYPE_AIGC, "ProjectName": project_name},
            access_key, secret_key,
        )
    except ArkActionError as e:
        # Duplicate-name race (two parallel runs both missed, both created) or
        # an eventually-consistent list: re-list once and adopt before failing.
        retry = await _find_exact()
        if retry is not None:
            print(f"[ByteplusAssetAPI] CreateAssetGroup({ref!r}) failed ({e.code}) but the group now exists; adopting {retry}.")
            return retry
        raise
    group_id = created.get("Id")
    if not group_id:
        raise RuntimeError(f"CreateAssetGroup({ref!r}) returned no Id: {created}")
    return str(group_id)


async def find_existing_asset_by_name(
    session: aiohttp.ClientSession,
    access_key: str,
    secret_key: str,
    group_id: str,
    name: str,
    *,
    asset_type: str,
) -> str | None:
    """Remote-adopt tier: return the asset Id for an exact (name, type, Active)
    match in the group, or None. Survives local-cache loss without duplicating
    the asset in the library."""
    async for item in _list_all_pages(
        session, "ListAssets",
        {
            # Filter.GroupType is REQUIRED by the API (empirical 2026-06-11:
            # MissingParameter.Filter.GroupType without it).
            "Filter": {"GroupIds": [group_id], "Name": name, "Statuses": ["Active"], "GroupType": GROUP_TYPE_AIGC},
        },
        access_key, secret_key,
    ):
        if item.get("Name") == name and item.get("AssetType") == asset_type and item.get("Status") == "Active":
            return str(item["Id"])
    return None


async def get_asset(
    session: aiohttp.ClientSession,
    access_key: str,
    secret_key: str,
    asset_id: str,
    *,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> dict[str, Any]:
    """GetAsset — authoritative status for one asset (100 QPS; cheap liveness check)."""
    return await action_call(
        session, "GetAsset", {"Id": asset_id, "ProjectName": project_name}, access_key, secret_key,
    )


async def create_asset(
    session: aiohttp.ClientSession,
    access_key: str,
    secret_key: str,
    *,
    group_id: str,
    url: str,
    asset_type: str,
    name: str,
    moderation_strategy: str = "Default",
    project_name: str = DEFAULT_PROJECT_NAME,
) -> str:
    """CreateAsset (async server-side) — returns the new asset Id. Caller polls."""
    body: dict[str, Any] = {
        "GroupId": group_id,
        "URL": url,
        "AssetType": asset_type,
        "Name": name,
        "ProjectName": project_name,
    }
    if moderation_strategy and moderation_strategy != "Default":
        # Skip requires the console content-pre-filter toggle to be off first.
        body["Moderation"] = {"Strategy": moderation_strategy}
    result = await action_call(session, "CreateAsset", body, access_key, secret_key)
    asset_id = result.get("Id")
    if not asset_id:
        raise RuntimeError(f"CreateAsset returned no Id: {result}")
    return str(asset_id)


async def poll_asset_active(
    session: aiohttp.ClientSession,
    access_key: str,
    secret_key: str,
    asset_id: str,
    *,
    timeout_s: float,
    interval_s: float = 5.0,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> str:
    """Poll GetAsset until Active/Failed or timeout. Returns final status string."""
    import time as _time

    deadline = _time.monotonic() + timeout_s
    status = "Processing"
    while _time.monotonic() < deadline:
        info = await get_asset(session, access_key, secret_key, asset_id, project_name=project_name)
        status = str(info.get("Status") or "Processing")
        if status in ("Active", "Failed"):
            return status
        await asyncio.sleep(interval_s)
    return f"Timeout({status})"


# ---------------------------------------------------------------------------
# Seedance 2.0 NATIVE asset-file constraints (BytePlus CreateAsset API reference,
# 2026-06-11). NOTE: these are NOT the Moyu constraints — the native video
# pixel-area cap is much higher (1080p fits: 1920x1080 = 2,073,600 < 2,086,876).
# Preflight here so a bad ref fails BEFORE the B2 stage + CreateAsset round-trip.
# ---------------------------------------------------------------------------

IMAGE_MIN_SIDE_PX = 300
IMAGE_MAX_SIDE_PX = 6000
IMAGE_MIN_ASPECT = 0.4     # W/H, exclusive bounds per docs "(0.4, 2.5)"
IMAGE_MAX_ASPECT = 2.5
IMAGE_MAX_BYTES = 30 * 1024 * 1024

VIDEO_MIN_DURATION_S = 2.0
VIDEO_MAX_DURATION_S = 15.0
# Lower bound relaxed to 23.0 for NTSC 24p (23.976), matching the empirically
# safe relaxation used on the Moyu side since 2026-05-28.
VIDEO_MIN_FPS = 23.0
VIDEO_MAX_FPS = 60.0
VIDEO_MIN_PIXEL_AREA = 409_600
VIDEO_MAX_PIXEL_AREA = 2_086_876
VIDEO_MAX_BYTES = 50 * 1024 * 1024


def preflight_image(image) -> tuple[bool, str]:  # noqa: ANN001 — torch tensor, kept untyped for import-light tests
    """Check IMAGE tensor dims + aspect against native CreateAsset constraints."""
    h = int(image.shape[-3])
    w = int(image.shape[-2])
    if not (IMAGE_MIN_SIDE_PX <= h <= IMAGE_MAX_SIDE_PX):
        return False, f"image height {h}px outside native range [{IMAGE_MIN_SIDE_PX}, {IMAGE_MAX_SIDE_PX}]. Resize before register."
    if not (IMAGE_MIN_SIDE_PX <= w <= IMAGE_MAX_SIDE_PX):
        return False, f"image width {w}px outside native range [{IMAGE_MIN_SIDE_PX}, {IMAGE_MAX_SIDE_PX}]. Resize before register."
    aspect = w / h
    if not (IMAGE_MIN_ASPECT < aspect < IMAGE_MAX_ASPECT):
        return False, (
            f"image aspect ratio W/H={aspect:.3f} outside native range "
            f"({IMAGE_MIN_ASPECT}, {IMAGE_MAX_ASPECT}). Crop/pad before register."
        )
    return True, f"ok ({w}x{h}, ar={aspect:.3f})"


def preflight_video(video) -> tuple[bool, str]:  # noqa: ANN001 — Input.Video duck-typed
    """Check VIDEO duration/fps/pixel-area against native CreateAsset constraints.

    Collects ALL violations so the user fixes everything in one pass.
    """
    violations: list[str] = []
    try:
        components = video.get_components()
        frames = components.images  # [F, H, W, C]
        h = int(frames.shape[1])
        w = int(frames.shape[2])
        pixel_area = h * w
        fps_val = float(components.frame_rate)
    except Exception as e:
        return False, f"could not inspect video components: {e.__class__.__name__}: {e}"

    try:
        duration = float(video.get_duration())
    except Exception:
        try:
            duration = frames.shape[0] / fps_val if fps_val else None
        except Exception:
            duration = None

    if duration is not None and not (VIDEO_MIN_DURATION_S <= duration <= VIDEO_MAX_DURATION_S):
        violations.append(f"duration {duration:.2f}s outside [{VIDEO_MIN_DURATION_S}, {VIDEO_MAX_DURATION_S}]s")
    if not (VIDEO_MIN_FPS <= fps_val <= VIDEO_MAX_FPS):
        violations.append(f"fps {fps_val:.2f} outside [{VIDEO_MIN_FPS}, {VIDEO_MAX_FPS}]")
    if not (VIDEO_MIN_PIXEL_AREA <= pixel_area <= VIDEO_MAX_PIXEL_AREA):
        msg = f"pixel area {pixel_area:,} ({w}x{h}) outside [{VIDEO_MIN_PIXEL_AREA:,}, {VIDEO_MAX_PIXEL_AREA:,}]"
        if pixel_area > VIDEO_MAX_PIXEL_AREA:
            msg += " — TOO LARGE. Native cap admits 1080p (1920x1080) but not above"
        else:
            msg += " — TOO SMALL. Resize to at least ~480p (854x480)"
        violations.append(msg)

    if violations:
        return False, "native ref-video constraint violation: " + "; ".join(violations)
    if duration is not None:
        return True, f"ok ({w}x{h}, {fps_val:.2f}fps, {duration:.2f}s)"
    return True, f"ok ({w}x{h}, {fps_val:.2f}fps)"


# ---------------------------------------------------------------------------
# B2 staging — EPHEMERAL upload hop for CreateAsset's URL-pull contract.
#
# Distinct from nv_b2_host_helpers (Moyu plane): own object-key prefix, and a
# delete-after-Active cleanup the Moyu path doesn't have (Moyu URLs stay
# relevant longer; here ByteDance copies the file into TOS at registration,
# so the staging object is garbage the moment the asset is Active).
# ---------------------------------------------------------------------------

B2_STAGING_PREFIX = "byteplus_asset_staging"
_B2_PRESIGN_EXPIRY_S = 3600  # ByteDance pulls within seconds; 1h is generous margin


def _b2_client(key_id: str, application_key: str, region: str):  # noqa: ANN001 — boto3 client type opaque
    try:
        import boto3  # type: ignore[import-not-found]
        from botocore.config import Config  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "boto3 is required for the B2 staging hop. Install with:\n  pip install boto3"
        ) from e
    return boto3.client(
        "s3",
        endpoint_url=f"https://s3.{region}.backblazeb2.com",
        aws_access_key_id=key_id,
        aws_secret_access_key=application_key,
        region_name=region,
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )


def stage_bytes_to_b2_sync(
    *,
    body: bytes,
    content_hash: str,
    ext: str,
    content_type: str,
    key_id: str,
    application_key: str,
    bucket: str,
    region: str,
) -> tuple[str, str]:
    """Upload bytes to the staging prefix; return (presigned_url, object_key).

    Content-hash object key = idempotent put: re-staging the same content
    overwrites the same key, never accumulates copies. Sync — call via
    asyncio.to_thread.
    """
    object_key = f"{B2_STAGING_PREFIX}/{content_hash}{ext}"
    client = _b2_client(key_id, application_key, region)
    client.put_object(Bucket=bucket, Key=object_key, Body=body, ContentType=content_type)
    url = client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": object_key}, ExpiresIn=_B2_PRESIGN_EXPIRY_S,
    )
    print(f"[ByteplusAssetAPI] staged s3://{bucket}/{object_key} ({len(body) // 1024} KB, presigned {_B2_PRESIGN_EXPIRY_S}s)")
    return url, object_key


def delete_staged_object_sync(
    *,
    object_key: str,
    key_id: str,
    application_key: str,
    bucket: str,
    region: str,
) -> bool:
    """Best-effort delete of a staging object after the asset reached Active.

    Returns True on success, False on any failure (logged, never raised —
    a leftover staging object is harmless; the registration already succeeded).
    """
    try:
        client = _b2_client(key_id, application_key, region)
        client.delete_object(Bucket=bucket, Key=object_key)
        print(f"[ByteplusAssetAPI] cleaned staging object s3://{bucket}/{object_key}")
        return True
    except Exception as e:
        print(f"[ByteplusAssetAPI] WARN: staging cleanup failed for {object_key}: {e.__class__.__name__}: {e}")
        return False
