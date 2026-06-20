"""BytePlus (native) asset library registration nodes — image + video.

Native sibling of `nv_moyu_asset_register.py` for the BytePlus ModelArk
trusted-asset library (Seedance 2.0 international plane). COMPLETELY SEPARATE
node group: zero imports from the Moyu helper/cache stack, own cache file,
own staging prefix — existing Moyu workflows are untouched by construction
(design constraint 2026-06-11).

Why this exists: direct base64/URL real-face refs are INPUT-GATED on the
native generation endpoint (InputImageSensitiveContentDetected.PrivacyInformation).
The bypass — runtime-validated end-to-end 2026-06-11 — is registering the ref
into the Virtual Portrait (AIGC) asset library and generating with
`asset://<id>`. These nodes do that registration console-free.

Upload/download hygiene contract (design requirements 2026-06-11):
1. Content-hash identity (SHA256) — same content NEVER uploads twice.
2. Three-tier lookup before any upload: local cache → remote ListAssets by
   deterministic name (`nv-ref-{hash[:24]}`) → fresh upload only if both miss.
   Known accepted risk (R0 review): two PARALLEL processes first-registering
   the SAME content can both miss both tiers and create a duplicate library
   asset (cosmetic; 1M quota; steady state self-corrects via remote adopt).
3. Idempotent B2 staging keyed by content hash; staging object deleted after
   the asset reaches Active (the file then lives in ByteDance TOS — B2
   steady-state footprint is zero).
4. Cache liveness self-heal: cached hits are verified via GetAsset (cheap,
   100 QPS) and stale entries fall through to re-register instead of
   returning dead asset ids (fixes the known Moyu-cache flaw).
5. The node registers exactly the ONE tensor wired into it — no folder scans,
   no batch side effects.

Flow per execute:
  hash → [verify-]cache_hit?  → return
       → remote adopt by name → cache + return
       → preflight → encode → B2 stage → CreateAsset → poll Active
       → cache + staging cleanup → return
"""

from __future__ import annotations

import asyncio
import io
import json
import time

import aiohttp

from comfy_api.latest import IO, Input

from . import nv_byteplus_asset_cache as cache
from .api_keys import resolve_b2_credentials
from .nv_byteplus_asset_api import (
    DEFAULT_ASSET_GROUP_NAME,
    DEFAULT_PROJECT_NAME,
    IMAGE_MAX_BYTES,
    VIDEO_MAX_BYTES,
    ArkActionError,
    create_asset,
    delete_staged_object_sync,
    deterministic_asset_name,
    ensure_asset_group,
    find_existing_asset_by_name,
    get_asset,
    poll_asset_active,
    preflight_image,
    preflight_video,
    resolve_ark_ak_sk,
    stage_bytes_to_b2_sync,
)

MODERATION_STRATEGIES = ["Default", "Skip"]

_BATCH_MAX_IMAGES = 9   # Seedance Mode C reference-image cap
_BATCH_MAX_VIDEOS = 3   # Seedance reference-video cap (matches the gen node)


# ---------------------------------------------------------------------------
# Local tensor → bytes encoders. Small local copies (NOT imported from the
# Moyu-plane b2 helpers) so the two node groups stay fully decoupled.
# ---------------------------------------------------------------------------


def _encode_image_to_png_bytes(image) -> bytes:  # noqa: ANN001 — torch tensor [1,H,W,C] or [H,W,C], float 0-1
    import numpy as np
    from PIL import Image as PILImage

    t = image
    if hasattr(t, "detach"):
        # .contiguous(): batch-sliced/transposed tensors raise
        # "ndarray is not C-contiguous" on .numpy() paths (R0 review).
        t = t.detach().contiguous().cpu()
    arr = t.numpy() if hasattr(t, "numpy") else t
    if arr.ndim == 4:
        arr = arr[0]
    arr = (arr.clip(0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if arr.shape[-1] == 4:
        pil_mode = "RGBA"
    else:
        pil_mode = "RGB"
        arr = arr[..., :3]
    buf = io.BytesIO()
    PILImage.fromarray(arr, mode=pil_mode).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _encode_video_to_mp4_bytes(video) -> bytes:  # noqa: ANN001 — Input.Video
    from comfy_api.latest import Types

    buf = io.BytesIO()
    video.save_to(buf, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
    buf.seek(0)
    return buf.getvalue()


def _image_bytes_for_upload(image) -> bytes:  # noqa: ANN001
    """Encode + enforce the native 30MB image cap. Raises ValueError over cap."""
    body = _encode_image_to_png_bytes(image)
    if len(body) > IMAGE_MAX_BYTES:
        raise ValueError(
            f"encoded PNG is {len(body) / (1024 * 1024):.1f}MB, exceeds native 30MB image cap. Downscale before register."
        )
    return body


def _video_bytes_for_upload(video) -> bytes:  # noqa: ANN001
    """Encode + enforce the native 50MB video cap. Raises ValueError over cap."""
    body = _encode_video_to_mp4_bytes(video)
    if len(body) > VIDEO_MAX_BYTES:
        raise ValueError(
            f"encoded MP4 is {len(body) / (1024 * 1024):.1f}MB, exceeds native 50MB video cap. Trim/downscale before register."
        )
    return body


# ---------------------------------------------------------------------------
# Shared registration core
# ---------------------------------------------------------------------------


async def _verify_cached_entry(
    session: aiohttp.ClientSession,
    ak: str,
    sk: str,
    cached: dict,
    *,
    project_name: str,
) -> tuple[bool, str]:
    """GetAsset liveness check for a cached entry. Returns (alive, detail).

    Transport/API errors are treated as ALIVE (don't block work on a flaky
    check — the generation call will surface a truly-dead asset loudly).
    Only an authoritative non-Active status or a NotFound-class error code
    declares the entry dead.
    """
    try:
        info = await get_asset(session, ak, sk, cached["asset_id"], project_name=project_name)
        status = str(info.get("Status") or "")
        if status == "Active":
            return True, "GetAsset says Active"
        return False, f"GetAsset says Status={status!r}"
    except ArkActionError as e:
        if "notfound" in e.code.lower() or "invalidparameter" in e.code.lower():
            return False, f"GetAsset error {e.code}: {e.message}"
        print(f"[ByteplusAssetRegister] WARN: liveness check inconclusive ({e.code}); trusting cache.")
        return True, f"liveness check inconclusive ({e.code}); trusted"
    except Exception as e:
        print(f"[ByteplusAssetRegister] WARN: liveness check transport error ({e.__class__.__name__}); trusting cache.")
        return True, f"liveness check transport error ({e.__class__.__name__}); trusted"


async def _register_core(
    *,
    node_tag: str,
    content_hash: str,
    get_body_bytes,  # () -> bytes; called ONLY on the fresh-upload tier, off the event loop
    file_ext: str,
    content_type: str,
    asset_type: str,
    tag: str,
    group_ref: str,
    project_name: str,
    moderation_strategy: str,
    force_reupload: bool,
    verify_cached: bool,
    cleanup_staging: bool,
    poll_timeout_s: float,
    ark_access_key: str,
    ark_secret_key: str,
    b2_key_id: str,
    b2_application_key: str,
    b2_bucket: str,
    b2_region: str,
) -> dict:
    """Three-tier register: cache → remote-adopt → B2-staged CreateAsset.

    `get_body_bytes` is invoked lazily (via asyncio.to_thread) only when the
    fresh-upload tier is actually reached — cache/remote hits never pay the
    PNG/MP4 encode. It may raise ValueError for post-encode size-cap
    violations; that surfaces as status='failed' like any other error.

    Returns a result dict: asset_id, asset_url, status (`cache_hit` /
    `remote_hit` / `uploaded` / `failed`), path, info, error, plus forensics.
    Never raises — callers decide raise-vs-soft via error_on_fail.
    """
    ak, sk = resolve_ark_ak_sk(ark_access_key, ark_secret_key)
    result: dict = {
        "asset_id": "",
        "asset_url": "",
        "status": "failed",
        "path": "",
        "info": "",
        "error": None,
        "content_hash": content_hash,
        "asset_type": asset_type,
        "tag": tag,
        "group_id": None,
        "group_ref": group_ref,
        "project_name": project_name,
        "self_healed": False,
        "staging_cleaned": None,
    }

    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=120)
    connector = aiohttp.TCPConnector(force_close=True, limit=2)
    session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    try:
        # Tier 1: local cache (+ optional liveness verification / self-heal)
        if not force_reupload:
            cached = cache.lookup(ak, content_hash)
            if cached is not None and (
                cached.get("asset_type") != asset_type
                or cached.get("project_name", DEFAULT_PROJECT_NAME) != project_name
            ):
                # Scope mismatch (R0 review): the cached asset is real but lives
                # in a different project (or is a different asset type, which
                # should be impossible by hash construction). Assets are
                # project-isolated at GENERATION time, so returning it would
                # produce a confusing downstream failure. Treat as a miss —
                # do NOT invalidate; the entry is still valid for its own scope.
                print(
                    f"[{node_tag}] cache entry scope mismatch "
                    f"(cached project={cached.get('project_name')!r}/type={cached.get('asset_type')!r} "
                    f"vs requested {project_name!r}/{asset_type!r}); treating as cache miss."
                )
                cached = None
            if cached is not None:
                if verify_cached:
                    alive, detail = await _verify_cached_entry(session, ak, sk, cached, project_name=project_name)
                    if alive:
                        cache.mark_verified(ak, content_hash)
                        result.update({
                            "asset_id": cached["asset_id"],
                            "asset_url": cached["asset_url"],
                            "status": "cache_hit",
                            "path": "local cache (liveness-verified)",
                            "info": f"Cached registration verified live ({detail}). No upload.",
                            "group_id": cached.get("group_id"),
                        })
                        return result
                    print(f"[{node_tag}] cached asset {cached['asset_id']} is DEAD ({detail}); self-healing (invalidate + re-register).")
                    cache.invalidate(ak, content_hash)
                    result["self_healed"] = True
                else:
                    result.update({
                        "asset_id": cached["asset_id"],
                        "asset_url": cached["asset_url"],
                        "status": "cache_hit",
                        "path": "local cache (unverified)",
                        "info": (
                            f"Returned from local cache without liveness check "
                            f"(verify_cached=False). registered_at={cached.get('registered_at')}."
                        ),
                        "group_id": cached.get("group_id"),
                    })
                    return result

        # Resolve group (id passthrough / name lookup / create-if-missing)
        group_id = await ensure_asset_group(session, ak, sk, group_ref, project_name=project_name)
        result["group_id"] = group_id

        # Tier 2: remote adopt by deterministic name (survives local-cache loss)
        target_name = deterministic_asset_name(content_hash)
        if not force_reupload:
            existing_id = await find_existing_asset_by_name(session, ak, sk, group_id, target_name, asset_type=asset_type)
            if existing_id is not None:
                asset_url = f"asset://{existing_id}"
                cache.register(
                    ak, content_hash,
                    asset_id=existing_id, asset_url=asset_url, asset_type=asset_type, tag=tag,
                    group_id=group_id, group_name=group_ref, project_name=project_name,
                )
                result.update({
                    "asset_id": existing_id,
                    "asset_url": asset_url,
                    "status": "remote_hit",
                    "path": "existing Active asset adopted by deterministic name",
                    "info": f"Found {existing_id} (name={target_name}) already Active in group {group_id}. Cached for future runs; no upload.",
                })
                return result

        # Tier 3: fresh upload — encode (lazy) → B2 stage → CreateAsset → poll → cleanup
        body_bytes = await asyncio.to_thread(get_body_bytes)
        print(f"[{node_tag}] no cache/remote hit; staging {len(body_bytes) // 1024} KB to B2 then CreateAsset (group={group_id}, name={target_name}).")
        k_id, k_app, k_bucket, k_region = resolve_b2_credentials(b2_key_id, b2_application_key, b2_bucket, b2_region)
        staged_url, object_key = await asyncio.to_thread(
            stage_bytes_to_b2_sync,
            body=body_bytes, content_hash=content_hash, ext=file_ext, content_type=content_type,
            key_id=k_id, application_key=k_app, bucket=k_bucket, region=k_region,
        )
        # Forensics: record the object KEY, not URL fragments (a presigned-URL
        # tail can leak part of the signature — R0 review).
        result["staging_object_key"] = object_key

        asset_id = await create_asset(
            session, ak, sk,
            group_id=group_id, url=staged_url, asset_type=asset_type, name=target_name,
            moderation_strategy=moderation_strategy, project_name=project_name,
        )
        print(f"[{node_tag}] CreateAsset accepted id={asset_id}; polling GetAsset for Active (timeout {poll_timeout_s:.0f}s)...")
        final_status = await poll_asset_active(session, ak, sk, asset_id, timeout_s=poll_timeout_s, project_name=project_name)

        if final_status != "Active":
            result.update({
                "asset_id": asset_id,
                "status": "failed",
                "path": f"upload reached terminal status={final_status}",
                "error": (
                    f"Asset {asset_id} did not reach Active (final: {final_status}). "
                    f"If Failed: content rejected at preprocessing (check the asset in the ModelArk console "
                    f"My assets view for the rejection reason; common causes are multi-face images in "
                    f"face-checked groups, content pre-filter hits, or unfetchable staging URL). "
                    f"If Timeout: the async ingest queue is slow — re-run later; the same content hash will "
                    f"adopt the asset if it eventually went Active."
                ),
            })
            # Leave the staging object in place on failure — it may still be
            # mid-pull, and it's the only evidence for debugging a rejection.
            result["staging_cleaned"] = False
            return result

        asset_url = f"asset://{asset_id}"
        cache.register(
            ak, content_hash,
            asset_id=asset_id, asset_url=asset_url, asset_type=asset_type, tag=tag,
            group_id=group_id, group_name=group_ref, project_name=project_name,
            source_url_tail=object_key,  # object key, not URL fragments (signature leak class)
        )
        if cleanup_staging:
            cleaned = await asyncio.to_thread(
                delete_staged_object_sync,
                object_key=object_key, key_id=k_id, application_key=k_app, bucket=k_bucket, region=k_region,
            )
            result["staging_cleaned"] = cleaned
        else:
            result["staging_cleaned"] = False

        result.update({
            "asset_id": asset_id,
            "asset_url": asset_url,
            "status": "uploaded",
            "path": "fresh upload via B2 staging + CreateAsset",
            "info": (
                f"Fresh registration. asset_id={asset_id}, group_id={group_id}, name={target_name}. "
                f"File now lives in ByteDance TOS; local cache updated — same content is free forever."
            ),
        })
        return result
    except Exception as e:
        result.update({
            "status": "failed",
            "path": "exception during registration",
            "error": f"{e.__class__.__name__}: {e}",
        })
        return result
    finally:
        await session.close()
        await asyncio.sleep(0.1)  # Windows ProactorEventLoop SSL cleanup tick


# ---------------------------------------------------------------------------
# Shared widget builders — both nodes expose the identical control surface
# ---------------------------------------------------------------------------


def _common_inputs(default_poll_timeout: float, max_poll_timeout: float) -> list:
    return [
        IO.String.Input(
            "tag",
            default="",
            tooltip=(
                "Optional human-friendly label saved with the local cache entry "
                "(e.g. 'jon_face_front'). Does NOT affect dedup — content hash is the only key."
            ),
            optional=True,
        ),
        IO.String.Input(
            "asset_group",
            default=DEFAULT_ASSET_GROUP_NAME,
            tooltip=(
                "Target Virtual Portrait (AIGC) asset group: either a group NAME "
                "(resolved via ListAssetGroups; created if missing) or a literal "
                "`group-...` id (used as-is). One group per character is the "
                "recommended organization. NOTE: first-ever group creation on an "
                "account requires the one-time authorization letter in the console "
                "(already signed on the studio account)."
            ),
            optional=True,
        ),
        IO.String.Input(
            "project_name",
            default=DEFAULT_PROJECT_NAME,
            tooltip=(
                "ModelArk project scope. Assets are project-isolated — the generation "
                "endpoint must live in the SAME project or asset:// refs fail."
            ),
            optional=True,
        ),
        IO.Combo.Input(
            "moderation",
            options=MODERATION_STRATEGIES,
            default="Default",
            tooltip=(
                "Content Pre-filter strategy for this asset. `Skip` bypasses most "
                "non-baseline review policies but requires the content pre-filter "
                "toggle to be turned off in the console first (enterprise Advanced "
                "Creation Rights). Leave `Default` unless a legitimate asset is "
                "being rejected by the pre-filter."
            ),
            optional=True,
        ),
        IO.Boolean.Input(
            "verify_cached",
            default=True,
            tooltip=(
                "On a local-cache hit, confirm the asset is still Active via GetAsset "
                "(cheap, 100 QPS) before returning it. A dead/deleted asset self-heals: "
                "cache entry invalidated, registration re-runs. Disable only to shave "
                "one API round-trip when you are certain the library hasn't changed."
            ),
            optional=True,
        ),
        IO.Boolean.Input(
            "force_reupload",
            default=False,
            tooltip=(
                "Skip the cache AND remote-adopt lookups, force a fresh upload. "
                "Creates a duplicate library asset if the content is already "
                "registered — use only for debugging registration itself."
            ),
            optional=True,
        ),
        IO.Boolean.Input(
            "cleanup_staging",
            default=True,
            tooltip=(
                "Delete the B2 staging object after the asset reaches Active (the "
                "file lives in ByteDance TOS from then on — staging copy is garbage). "
                "Disable to keep the staged file for debugging. On registration "
                "FAILURE the staging object is always kept as evidence."
            ),
            optional=True,
        ),
        IO.Float.Input(
            "poll_timeout_s",
            default=default_poll_timeout,
            min=30.0,
            max=max_poll_timeout,
            step=10.0,
            tooltip=(
                "Max seconds to wait for the new asset to reach Active. Only applies "
                "to the fresh-upload path. CreateAsset is async server-side with no "
                "SLA; video ingestion is documented as slower than image."
            ),
            optional=True,
        ),
        IO.Boolean.Input(
            "error_on_fail",
            default=False,
            tooltip=(
                "If True, raise on registration failure (fail-fast pipelines). If "
                "False (default), return empty asset_url + status='failed' + error "
                "details in info_json so downstream nodes can branch."
            ),
            optional=True,
        ),
        IO.String.Input(
            "ark_access_key",
            default="",
            tooltip="Optional ARK Access Key override. Empty → env ARK_ACCESS_KEY / .env. (IAM AK/SK pair, NOT the Bearer API key.)",
            optional=True,
        ),
        IO.String.Input(
            "ark_secret_key",
            default="",
            tooltip="Optional ARK Secret Key override. Empty → env ARK_SECRET_KEY / .env.",
            optional=True,
        ),
        IO.String.Input(
            "b2_key_id",
            default="",
            tooltip="B2 Application Key ID for the staging hop. Empty → env B2_KEY_ID / .env.",
            optional=True,
        ),
        IO.String.Input(
            "b2_application_key",
            default="",
            tooltip="B2 Application Key. Empty → env B2_APPLICATION_KEY / .env.",
            optional=True,
        ),
        IO.String.Input(
            "b2_bucket",
            default="",
            tooltip="B2 staging bucket. Empty → env B2_BUCKET / default ``.",
            optional=True,
        ),
        IO.String.Input(
            "b2_region",
            default="",
            tooltip="B2 region. Empty → env B2_REGION / default `us-east-005`.",
            optional=True,
        ),
    ]


_OUTPUTS = [
    IO.String.Output(display_name="asset_url"),
    IO.String.Output(display_name="asset_id"),
    IO.String.Output(display_name="status"),
    IO.String.Output(display_name="info_json"),
]


def _finish(node_tag: str, result: dict, t_start: float, error_on_fail: bool) -> IO.NodeOutput:
    result["elapsed_seconds"] = round(time.time() - t_start, 2)
    print(f"[{node_tag}] {result['status'].upper()} via {result['path']} in {result['elapsed_seconds']}s. asset_url={result['asset_url']!r}")
    if result.get("error"):
        print(f"[{node_tag}] ERROR: {result['error']}")
        if error_on_fail:
            raise RuntimeError(f"[{node_tag}] {result['error']}")
    return IO.NodeOutput(
        result["asset_url"],
        result["asset_id"],
        result["status"],
        json.dumps(result, indent=2, ensure_ascii=False),
    )


def _preflight_fail(node_tag: str, msg: str, tag: str, error_on_fail: bool) -> IO.NodeOutput:
    err = f"[{node_tag}] PREFLIGHT FAILED: {msg}"
    print(err)
    if error_on_fail:
        raise RuntimeError(err)
    return IO.NodeOutput(
        "", "", "failed",
        json.dumps({"status": "failed", "path": "preflight rejected before upload", "error": msg, "tag": tag}, indent=2, ensure_ascii=False),
    )


def _batch_fail(node_tag: str, reason: str, error_on_fail: bool, results: list | None = None) -> IO.NodeOutput:
    """Uniform batch-failure exit (R0 review: honor error_on_fail for ALL
    failures, including upfront structural ones). Raises when error_on_fail
    (atomic default); else emits ('', 0, status_json) — never a partial URL list."""
    msg = f"[{node_tag}] {reason}"
    print(msg)
    if error_on_fail:
        raise ValueError(msg)
    if results is None:
        results = [{"slot": -1, "status": "failed", "error": reason}]
    return IO.NodeOutput("", 0, json.dumps(results, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Zerogen_ByteplusImageAssetRegister
# ---------------------------------------------------------------------------


class Zerogen_ByteplusImageAssetRegister(IO.ComfyNode):
    """Register an IMAGE into the BytePlus Virtual Portrait (AIGC) asset library.

    Wire a reference image in ONCE per character/outfit. Output `asset_url`
    (`asset://asset-...`) plugs into native Seedance generation content as a
    `reference_image` — the validated bypass for the real-face input gate.

    Hygiene: content-hash dedup, remote-adopt, liveness self-heal, ephemeral
    B2 staging with cleanup. Re-runs with the same image are free.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Zerogen_ByteplusImageAssetRegister",
            display_name="BytePlus Image Asset Register",
            category="zerogen",
            description=(
                "Register an IMAGE into the BytePlus (native) Virtual Portrait "
                "asset library, console-free. Content-hash dedup: local cache → "
                "remote adopt by deterministic name → fresh upload only if both "
                "miss. B2 is an ephemeral staging hop (deleted after Active). "
                "Output asset_url is the asset:// ref for native Seedance "
                "generation — the validated real-face bypass."
            ),
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip=(
                        "Reference image to register. SHA256-hashed for stable dedup "
                        "across sessions — same content always resolves to the same "
                        "library asset, never re-uploads. Native constraints: "
                        "300-6000px per side, W/H aspect in (0.4, 2.5), <30MB encoded."
                    ),
                ),
                *_common_inputs(default_poll_timeout=180.0, max_poll_timeout=600.0),
            ],
            outputs=_OUTPUTS,
            is_api_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, image=None, force_reupload: bool = False, project_name: str = DEFAULT_PROJECT_NAME, ark_access_key: str = "", ark_secret_key: str = "", **kwargs):  # noqa: ANN001, ANN206
        """V3 IS_CHANGED — make ComfyUI's execution cache failure-safe.

        R0 review convergent HIGH: with a plain input-signature cache, a FAILED
        run (empty asset_url, error_on_fail=False) would be cached and served
        forever, blocking recovery. Failed runs never write the local asset
        cache, so the partition is:
        - local cache HAS the content (in the requested project scope) → stable
          fingerprint (ComfyUI may skip re-execution; same content = same asset
          URL, no downstream cascade)
        - local cache MISSING / scope mismatch (never registered, prior failure,
          self-heal invalidated, or different project — R1 Codex) → NaN →
          always re-execute until a success lands in this scope.
        """
        try:
            if force_reupload or image is None:
                return float("nan")
            ak, _ = resolve_ark_ak_sk(ark_access_key, ark_secret_key)
            cached = cache.lookup(ak, cache.hash_image_tensor(image))
            if (
                cached is not None
                and cached.get("asset_type") == "Image"
                and cached.get("project_name", DEFAULT_PROJECT_NAME) == project_name
            ):
                return f"registered:{cached['asset_id']}:{project_name}"
        except Exception:
            pass  # any doubt → re-execute; the real error surfaces in execute()
        return float("nan")

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        tag: str = "",
        asset_group: str = DEFAULT_ASSET_GROUP_NAME,
        project_name: str = DEFAULT_PROJECT_NAME,
        moderation: str = "Default",
        verify_cached: bool = True,
        force_reupload: bool = False,
        cleanup_staging: bool = True,
        poll_timeout_s: float = 180.0,
        error_on_fail: bool = False,
        ark_access_key: str = "",
        ark_secret_key: str = "",
        b2_key_id: str = "",
        b2_application_key: str = "",
        b2_bucket: str = "",
        b2_region: str = "",
    ) -> IO.NodeOutput:
        t_start = time.time()
        node_tag = "Zerogen_ByteplusImageAssetRegister"

        # Batch guard: this single-asset node's encoder takes frame 0 only, but
        # the cache key hashes the whole tensor — feeding a [N>1,...] batch would
        # silently register just frame 0. Fail loudly and point at the batch node.
        if hasattr(image, "shape") and image.ndim == 4 and int(image.shape[0]) > 1:
            return _preflight_fail(
                node_tag,
                f"received a batch of {int(image.shape[0])} images; this single-asset node would register only frame 0. "
                f"Use Zerogen_ByteplusImageBatchRegister for multi-image registration.",
                tag, error_on_fail,
            )

        ok, msg = preflight_image(image)
        if not ok:
            return _preflight_fail(node_tag, msg, tag, error_on_fail)
        print(f"[{node_tag}] preflight: {msg}")

        content_hash = cache.hash_image_tensor(image)
        print(f"[{node_tag}] content_hash={content_hash[:16]}... tag={tag!r} group={asset_group!r}")

        result = await _register_core(
            node_tag=node_tag,
            content_hash=content_hash,
            get_body_bytes=lambda: _image_bytes_for_upload(image),
            file_ext=".png",
            content_type="image/png",
            asset_type="Image",
            tag=tag,
            group_ref=asset_group,
            project_name=project_name,
            moderation_strategy=moderation,
            force_reupload=force_reupload,
            verify_cached=verify_cached,
            cleanup_staging=cleanup_staging,
            poll_timeout_s=poll_timeout_s,
            ark_access_key=ark_access_key,
            ark_secret_key=ark_secret_key,
            b2_key_id=b2_key_id,
            b2_application_key=b2_application_key,
            b2_bucket=b2_bucket,
            b2_region=b2_region,
        )
        return _finish(node_tag, result, t_start, error_on_fail)


# ---------------------------------------------------------------------------
# Zerogen_ByteplusVideoAssetRegister
# ---------------------------------------------------------------------------


class Zerogen_ByteplusVideoAssetRegister(IO.ComfyNode):
    """Register a VIDEO into the BytePlus Virtual Portrait (AIGC) asset library.

    Same three-tier hygiene as the image variant. The MP4 encode is HOISTED
    behind the cache lookup — cache hits never pay the multi-second re-encode.

    First runtime use doubles as the empirical validation of video-asset
    registration on the native plane (image path validated 2026-06-11;
    video documented-but-untested).
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Zerogen_ByteplusVideoAssetRegister",
            display_name="BytePlus Video Asset Register",
            category="zerogen",
            description=(
                "Register a VIDEO into the BytePlus (native) asset library, "
                "console-free. Same content-hash dedup + B2 ephemeral staging as "
                "the image variant. Native video constraints: mp4/mov, 2-15s, "
                "fps 24-60, total pixels 409,600-2,086,876 (1080p fits), ≤50MB."
            ),
            inputs=[
                IO.Video.Input(
                    "video",
                    tooltip=(
                        "Reference video to register. Content signature = first frame "
                        "+ last frame + frame count (stable dedup without hashing every "
                        "frame). Native constraints: 2-15s, fps 24-60, pixel area "
                        "409,600-2,086,876 (1080p = 2,073,600 fits), ≤50MB encoded."
                    ),
                ),
                *_common_inputs(default_poll_timeout=360.0, max_poll_timeout=900.0),
            ],
            outputs=_OUTPUTS,
            is_api_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, video=None, force_reupload: bool = False, project_name: str = DEFAULT_PROJECT_NAME, ark_access_key: str = "", ark_secret_key: str = "", **kwargs):  # noqa: ANN001, ANN206
        """V3 IS_CHANGED — failure-safe, project-scoped (see image sibling)."""
        try:
            if force_reupload or video is None:
                return float("nan")
            ak, _ = resolve_ark_ak_sk(ark_access_key, ark_secret_key)
            cached = cache.lookup(ak, cache.hash_video_input(video))
            if (
                cached is not None
                and cached.get("asset_type") == "Video"
                and cached.get("project_name", DEFAULT_PROJECT_NAME) == project_name
            ):
                return f"registered:{cached['asset_id']}:{project_name}"
        except Exception:
            pass  # any doubt → re-execute; the real error surfaces in execute()
        return float("nan")

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        tag: str = "",
        asset_group: str = DEFAULT_ASSET_GROUP_NAME,
        project_name: str = DEFAULT_PROJECT_NAME,
        moderation: str = "Default",
        verify_cached: bool = True,
        force_reupload: bool = False,
        cleanup_staging: bool = True,
        poll_timeout_s: float = 360.0,
        error_on_fail: bool = False,
        ark_access_key: str = "",
        ark_secret_key: str = "",
        b2_key_id: str = "",
        b2_application_key: str = "",
        b2_bucket: str = "",
        b2_region: str = "",
    ) -> IO.NodeOutput:
        t_start = time.time()
        node_tag = "Zerogen_ByteplusVideoAssetRegister"

        ok, msg = preflight_video(video)
        if not ok:
            return _preflight_fail(node_tag, msg, tag, error_on_fail)
        print(f"[{node_tag}] preflight: {msg}")

        content_hash = cache.hash_video_input(video)
        print(f"[{node_tag}] content_hash={content_hash[:16]}... tag={tag!r} group={asset_group!r}")

        # The MP4 encode is lazy inside the core (Tier 3 only) — cache and
        # remote-adopt hits never pay the multi-second re-encode.
        result = await _register_core(
            node_tag=node_tag,
            content_hash=content_hash,
            get_body_bytes=lambda: _video_bytes_for_upload(video),
            file_ext=".mp4",
            content_type="video/mp4",
            asset_type="Video",
            tag=tag,
            group_ref=asset_group,
            project_name=project_name,
            moderation_strategy=moderation,
            force_reupload=force_reupload,
            verify_cached=verify_cached,
            cleanup_staging=cleanup_staging,
            poll_timeout_s=poll_timeout_s,
            ark_access_key=ark_access_key,
            ark_secret_key=ark_secret_key,
            b2_key_id=b2_key_id,
            b2_application_key=b2_application_key,
            b2_bucket=b2_bucket,
            b2_region=b2_region,
        )
        return _finish(node_tag, result, t_start, error_on_fail)


# ---------------------------------------------------------------------------
# Zerogen_ByteplusImageBatchRegister — register an IMAGE BATCH (1-9) as native
# assets in one shot. Handles the single-image case too (batch_size=1).
#
# BytePlus best practice: multiple assets of the SAME person go into ONE group
# (full-body + facial close-up + angles) to improve identity consistency at
# generation. This node registers each slot into the shared asset_group via the
# same reviewed `_register_core` the single node uses, and emits a newline-
# joined string of asset:// URLs in tensor batch order — ready to feed a native
# gen node's reference_image list (prompt refers to them as "image 1", "image
# 2", ... by request-body order).
#
# Atomic K==N semantics (duplicate-first from MoyuImageBatchRegister): all
# slots succeed or the batch fails as a unit; never emits K<N URLs.
# ---------------------------------------------------------------------------


class Zerogen_ByteplusImageBatchRegister(IO.ComfyNode):
    """Register an IMAGE BATCH (1-9) into the BytePlus asset library in one shot.

    Each slot runs through the same three-tier core as the single node
    (cache → remote adopt → B2-staged CreateAsset), into the SAME asset_group.
    Emits newline-joined asset:// URLs in batch order for a native gen node's
    reference list. Handles batch_size=1 identically, so it covers the single
    case — the standalone single node stays for simple one-ref graphs.

    Atomic: all slots succeed or the batch fails as a unit (never emits K<N
    URLs — that would desync a downstream "image N" prompt mapping). Per-slot
    content-hash dedup means rearranging the batch still hits cache.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Zerogen_ByteplusImageBatchRegister",
            display_name="BytePlus Image Batch Register",
            category="zerogen",
            description=(
                "Register an IMAGE BATCH (1-9) into the BytePlus (native) asset "
                "library in one shot — all slots into one asset_group (BytePlus "
                "best practice for same-person refs). Atomic: all succeed or the "
                "batch fails (never emits K<N URLs). Output joined_urls is the "
                "newline-separated asset:// list in batch order for a native gen "
                "node's reference_image list. Handles batch_size=1 too."
            ),
            inputs=[
                IO.Image.Input(
                    "images",
                    tooltip=(
                        "Batch of 1-9 reference images [B,H,W,C]. Tensor batch order "
                        "IS the registration order — slot 0 → line 0 of joined_urls, "
                        "etc. Each slot gets an independent content hash + cache lookup "
                        "(rearranging still hits cache). Native per-image constraints: "
                        "300-6000px sides, W/H aspect (0.4, 2.5), <30MB encoded."
                    ),
                ),
                IO.String.Input(
                    "tag_prefix",
                    default="",
                    tooltip=(
                        "Tag prefix. Per-slot tags auto-generated as "
                        "'{prefix}_{content_hash[:8]}' — content-stable across batch "
                        "reorderings. Empty → tags become 'batch_{content_hash[:8]}'. "
                        "Overridden by `tags` when present. Tags do NOT affect dedup."
                    ),
                    optional=True,
                ),
                IO.String.Input(
                    "tags",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Optional explicit per-slot tags, one per line "
                        "(e.g. 'fullbody\\nface_closeup\\nprofile'). Count MUST match "
                        "the images batch size exactly or the node raises before any "
                        "upload. Empty → auto tags via tag_prefix + content_hash."
                    ),
                    optional=True,
                ),
                # _common_inputs[0] is the single-asset `tag` widget — drop it;
                # the batch node uses tag_prefix/tags for per-slot labels.
                *_common_inputs(default_poll_timeout=180.0, max_poll_timeout=600.0)[1:],
            ],
            outputs=[
                IO.String.Output(display_name="joined_urls"),
                IO.Int.Output(display_name="count"),
                IO.String.Output(display_name="per_slot_status_json"),
            ],
            is_api_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, images=None, force_reupload: bool = False, project_name: str = DEFAULT_PROJECT_NAME, ark_access_key: str = "", ark_secret_key: str = "", **kwargs):  # noqa: ANN001, ANN206
        """V3 IS_CHANGED — failure-safe + project-scoped, ALL-slots-or-rerun.

        Stable fingerprint only when EVERY slot is already cached in the
        requested project scope; any uncached/mismatched/failed slot → NaN →
        re-execute (atomic with the node's all-or-nothing semantics).
        """
        try:
            if force_reupload or images is None or not hasattr(images, "shape") or images.ndim != 4:
                return float("nan")
            n = int(images.shape[0])
            if n < 1 or n > _BATCH_MAX_IMAGES:
                return float("nan")  # invalid batch size → re-execute (execute() will reject it)
            ak, _ = resolve_ark_ak_sk(ark_access_key, ark_secret_key)
            ids = []
            for i in range(n):
                cached = cache.lookup(ak, cache.hash_image_tensor(images[i:i + 1]))
                if not (
                    cached is not None
                    and cached.get("asset_type") == "Image"
                    and cached.get("project_name", DEFAULT_PROJECT_NAME) == project_name
                ):
                    return float("nan")
                ids.append(cached["asset_id"])
            return "batch:" + ":".join(ids) + f":{project_name}"
        except Exception:
            pass
        return float("nan")

    @classmethod
    async def execute(
        cls,
        images: Input.Image,
        tag_prefix: str = "",
        tags: str = "",
        asset_group: str = DEFAULT_ASSET_GROUP_NAME,
        project_name: str = DEFAULT_PROJECT_NAME,
        moderation: str = "Default",
        verify_cached: bool = True,
        force_reupload: bool = False,
        cleanup_staging: bool = True,
        poll_timeout_s: float = 180.0,
        error_on_fail: bool = True,
        ark_access_key: str = "",
        ark_secret_key: str = "",
        b2_key_id: str = "",
        b2_application_key: str = "",
        b2_bucket: str = "",
        b2_region: str = "",
    ) -> IO.NodeOutput:
        t_overall = time.time()
        node_tag = "Zerogen_ByteplusImageBatchRegister"

        # --- Phase 1: cheap deterministic preflights upfront (no network) ---
        # All failures route through _batch_fail so error_on_fail is honored
        # uniformly (R0 review HIGH): default True raises; False → ('', 0, json).
        if images is None or not hasattr(images, "shape"):
            return _batch_fail(node_tag, "`images` is missing or not a tensor.", error_on_fail)
        if images.ndim != 4:
            return _batch_fail(node_tag, f"`images` must be a 4D tensor [B,H,W,C]; got shape {tuple(images.shape)} (ndim={images.ndim}).", error_on_fail)
        batch_size = int(images.shape[0])
        if batch_size < 1:
            return _batch_fail(node_tag, "`images` batch is empty.", error_on_fail)
        if batch_size > _BATCH_MAX_IMAGES:
            return _batch_fail(node_tag, f"Seedance accepts at most {_BATCH_MAX_IMAGES} reference images; got batch size {batch_size}.", error_on_fail)

        explicit_tags = [t.strip() for t in (tags or "").splitlines() if t.strip()]
        if explicit_tags and len(explicit_tags) != batch_size:
            return _batch_fail(
                node_tag,
                f"`tags` has {len(explicit_tags)} non-empty lines but `images` batch has {batch_size} slots. "
                f"Provide zero tags (auto-named) or exactly {batch_size}, one per slot in batch order.",
                error_on_fail,
            )

        # Resolve AK early so a missing-cred error fails before any work.
        try:
            resolve_ark_ak_sk(ark_access_key, ark_secret_key)
        except Exception as e:
            return _batch_fail(node_tag, f"credential resolution failed: {e}", error_on_fail)

        per_slot: list[dict] = []
        for i in range(batch_size):
            single = images[i:i + 1]                       # preserve [1,H,W,C]
            content_hash = cache.hash_image_tensor(single)
            if explicit_tags:
                tag_i = explicit_tags[i]
            elif tag_prefix:
                tag_i = f"{tag_prefix}_{content_hash[:8]}"
            else:
                tag_i = f"batch_{content_hash[:8]}"
            per_slot.append({"slot": i, "image": single, "content_hash": content_hash, "tag": tag_i})

        # Per-slot image preflight — compute once per slot, collect ALL failures.
        slot_checks = [(s, *preflight_image(s["image"])) for s in per_slot]  # (s, ok, msg)
        preflight_failures = [(s["slot"], msg) for s, ok, msg in slot_checks if not ok]
        if preflight_failures:
            results = [
                {"slot": s["slot"], "content_hash": s["content_hash"], "tag": s["tag"],
                 "status": "preflight_failed",
                 "error": next((m for sl, m in preflight_failures if sl == s["slot"]), None)}
                for s in per_slot
            ]
            err = (
                f"native preflight failed for {len(preflight_failures)} of {batch_size} images:\n"
                + "\n".join(f"  - slot {s}: {m}" for s, m in preflight_failures)
            )
            return _batch_fail(node_tag, err, error_on_fail, results=results)

        # --- Phase 2: serial per-slot register; stop on first failure ---
        results: list[dict] = []
        first_failure_slot: int | None = None
        first_failure_error: str | None = None

        for s in per_slot:
            slot_t = time.time()
            core = await _register_core(
                node_tag=f"{node_tag}[slot {s['slot']}]",
                content_hash=s["content_hash"],
                get_body_bytes=(lambda img=s["image"]: _image_bytes_for_upload(img)),
                file_ext=".png",
                content_type="image/png",
                asset_type="Image",
                tag=s["tag"],
                group_ref=asset_group,
                project_name=project_name,
                moderation_strategy=moderation,
                force_reupload=force_reupload,
                verify_cached=verify_cached,
                cleanup_staging=cleanup_staging,
                poll_timeout_s=poll_timeout_s,
                ark_access_key=ark_access_key,
                ark_secret_key=ark_secret_key,
                b2_key_id=b2_key_id,
                b2_application_key=b2_application_key,
                b2_bucket=b2_bucket,
                b2_region=b2_region,
            )
            core["slot"] = s["slot"]
            core["elapsed_seconds"] = round(time.time() - slot_t, 3)
            results.append(core)
            if core.get("status") == "failed":
                first_failure_slot = s["slot"]
                first_failure_error = core.get("error") or "register returned status=failed"
                break

        # --- Phase 3: atomic assembly ---
        if first_failure_slot is not None:
            attempted = {r["slot"] for r in results}
            for s in per_slot:
                if s["slot"] not in attempted:
                    results.append({
                        "slot": s["slot"], "content_hash": s["content_hash"], "tag": s["tag"],
                        "status": "unattempted", "asset_url": "", "asset_id": "",
                        "error": f"batch stopped after slot {first_failure_slot} failed",
                    })
            results.sort(key=lambda r: r["slot"])
            overall = (
                f"[{node_tag}] BATCH FAILED at slot {first_failure_slot} of {batch_size}: "
                f"{first_failure_error}. Atomic semantics: emitting empty joined_urls (count=0)."
            )
            print(overall)
            if error_on_fail:
                raise RuntimeError(overall)
            return IO.NodeOutput("", 0, json.dumps(results, indent=2, ensure_ascii=False))

        urls = [r["asset_url"] for r in results]
        joined = "\n".join(urls)
        elapsed = round(time.time() - t_overall, 2)
        print(
            f"[{node_tag}] BATCH OK — {batch_size} slot(s) in {elapsed}s "
            f"({sum(1 for r in results if r['status'] == 'cache_hit')} cache_hit, "
            f"{sum(1 for r in results if r['status'] == 'remote_hit')} remote_hit, "
            f"{sum(1 for r in results if r['status'] == 'uploaded')} uploaded)."
        )
        return IO.NodeOutput(joined, batch_size, json.dumps(results, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Zerogen_ByteplusVideoBatchRegister — register up to 3 VIDEOS as native assets
# in one shot. The video sibling of Zerogen_ByteplusImageBatchRegister.
#
# Videos can't share one socket the way images do (an IMAGE is a [B,H,W,C] tensor;
# a VIDEO is a single object), so each ref is a discrete optional slot. Same
# content-hash dedup + B2 ephemeral staging + atomic K==N semantics. joined_urls
# (newline asset:// in slot order) feeds straight into the gen node's
# `ref_video_asset_urls` multiline input (@Video1..N order).
# ---------------------------------------------------------------------------


class Zerogen_ByteplusVideoBatchRegister(IO.ComfyNode):
    """Register up to 3 VIDEOS into the BytePlus asset library in one shot.

    Atomic: all slots succeed or the batch fails as a unit (never emits K<N URLs).
    Each slot reuses the same reviewed `_register_core` as the single video node
    (cache → remote-adopt → B2 upload), so cache hits never pay the re-encode.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Zerogen_ByteplusVideoBatchRegister",
            display_name="BytePlus Video Batch Register",
            category="zerogen",
            description=(
                "Register 1-3 reference VIDEOS into the BytePlus (native) asset "
                "library in one shot. Discrete slots (videos can't batch into one "
                "socket like images). Same content-hash dedup + B2 staging as the "
                "single node. Atomic: all succeed or the batch fails (never K<N). "
                "joined_urls is the newline-separated asset:// list in slot order — "
                "wire into Zerogen_ByteplusSeedanceGen.ref_video_asset_urls. Native "
                "per-video constraints: 2-15s, fps 24-60, pixel area 409,600-2,086,876, <=50MB."
            ),
            inputs=[
                IO.Video.Input(
                    "video_1",
                    tooltip="First reference video (required). Slot order IS @Video1..N order in joined_urls.",
                ),
                IO.Video.Input(
                    "video_2",
                    tooltip="Second reference video (optional).",
                    optional=True,
                ),
                IO.Video.Input(
                    "video_3",
                    tooltip="Third reference video (optional). Seedance accepts up to 3 ref videos.",
                    optional=True,
                ),
                IO.String.Input(
                    "tag_prefix",
                    default="",
                    tooltip=(
                        "Tag prefix. Per-slot tags auto-generated as "
                        "'{prefix}_{content_hash[:8]}' — content-stable. Empty → "
                        "'batch_{content_hash[:8]}'. Overridden by `tags`. Tags do NOT affect dedup."
                    ),
                    optional=True,
                ),
                IO.String.Input(
                    "tags",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Optional explicit per-slot tags, one per line. Count MUST match "
                        "the number of connected video slots exactly or the node raises "
                        "before any upload. Empty → auto tags via tag_prefix + content_hash."
                    ),
                    optional=True,
                ),
                # _common_inputs[0] is the single-asset `tag` widget — drop it;
                # the batch node uses tag_prefix/tags for per-slot labels.
                *_common_inputs(default_poll_timeout=360.0, max_poll_timeout=900.0)[1:],
            ],
            outputs=[
                IO.String.Output(display_name="joined_urls"),
                IO.Int.Output(display_name="count"),
                IO.String.Output(display_name="per_slot_status_json"),
            ],
            is_api_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, video_1=None, video_2=None, video_3=None, force_reupload: bool = False, project_name: str = DEFAULT_PROJECT_NAME, ark_access_key: str = "", ark_secret_key: str = "", **kwargs):  # noqa: ANN001, ANN206
        """V3 IS_CHANGED — stable only when EVERY connected slot is already cached
        in the requested project scope; any uncached slot → NaN → re-execute
        (atomic with the all-or-nothing semantics). Mirrors the image-batch sibling."""
        try:
            videos = [v for v in (video_1, video_2, video_3) if v is not None]
            if force_reupload or not videos:
                return float("nan")
            ak, _ = resolve_ark_ak_sk(ark_access_key, ark_secret_key)
            ids = []
            for v in videos:
                cached = cache.lookup(ak, cache.hash_video_input(v))
                if not (
                    cached is not None
                    and cached.get("asset_type") == "Video"
                    and cached.get("project_name", DEFAULT_PROJECT_NAME) == project_name
                ):
                    return float("nan")
                ids.append(cached["asset_id"])
            return "vbatch:" + ":".join(ids) + f":{project_name}"
        except Exception:
            pass
        return float("nan")

    @classmethod
    async def execute(
        cls,
        video_1: Input.Video,
        video_2: Input.Video = None,
        video_3: Input.Video = None,
        tag_prefix: str = "",
        tags: str = "",
        asset_group: str = DEFAULT_ASSET_GROUP_NAME,
        project_name: str = DEFAULT_PROJECT_NAME,
        moderation: str = "Default",
        verify_cached: bool = True,
        force_reupload: bool = False,
        cleanup_staging: bool = True,
        poll_timeout_s: float = 360.0,
        error_on_fail: bool = True,
        ark_access_key: str = "",
        ark_secret_key: str = "",
        b2_key_id: str = "",
        b2_application_key: str = "",
        b2_bucket: str = "",
        b2_region: str = "",
    ) -> IO.NodeOutput:
        t_overall = time.time()
        node_tag = "Zerogen_ByteplusVideoBatchRegister"

        # --- Phase 1: cheap deterministic preflights upfront (no network) ---
        videos = [v for v in (video_1, video_2, video_3) if v is not None]
        n = len(videos)
        if n < 1:
            return _batch_fail(node_tag, "no video connected (at least video_1 is required).", error_on_fail)
        if n > _BATCH_MAX_VIDEOS:
            return _batch_fail(node_tag, f"Seedance accepts at most {_BATCH_MAX_VIDEOS} reference videos; got {n}.", error_on_fail)

        explicit_tags = [t.strip() for t in (tags or "").splitlines() if t.strip()]
        if explicit_tags and len(explicit_tags) != n:
            return _batch_fail(
                node_tag,
                f"`tags` has {len(explicit_tags)} non-empty lines but {n} video slot(s) are connected. "
                f"Provide zero tags (auto-named) or exactly {n}, one per slot in order.",
                error_on_fail,
            )

        # Resolve AK early so a missing-cred error fails before any work.
        try:
            resolve_ark_ak_sk(ark_access_key, ark_secret_key)
        except Exception as e:
            return _batch_fail(node_tag, f"credential resolution failed: {e}", error_on_fail)

        per_slot: list[dict] = []
        for i, vid in enumerate(videos):
            content_hash = cache.hash_video_input(vid)
            if explicit_tags:
                tag_i = explicit_tags[i]
            elif tag_prefix:
                tag_i = f"{tag_prefix}_{content_hash[:8]}"
            else:
                tag_i = f"batch_{content_hash[:8]}"
            per_slot.append({"slot": i, "video": vid, "content_hash": content_hash, "tag": tag_i})

        # Per-slot video preflight — collect ALL failures upfront.
        slot_checks = [(s, *preflight_video(s["video"])) for s in per_slot]  # (s, ok, msg)
        preflight_failures = [(s["slot"], msg) for s, ok, msg in slot_checks if not ok]
        if preflight_failures:
            results = [
                {"slot": s["slot"], "content_hash": s["content_hash"], "tag": s["tag"],
                 "status": "preflight_failed",
                 "error": next((m for sl, m in preflight_failures if sl == s["slot"]), None)}
                for s in per_slot
            ]
            err = (
                f"native preflight failed for {len(preflight_failures)} of {n} videos:\n"
                + "\n".join(f"  - slot {s}: {m}" for s, m in preflight_failures)
            )
            return _batch_fail(node_tag, err, error_on_fail, results=results)

        # --- Phase 2: serial per-slot register; stop on first failure ---
        results: list[dict] = []
        first_failure_slot: int | None = None
        first_failure_error: str | None = None

        for s in per_slot:
            slot_t = time.time()
            core = await _register_core(
                node_tag=f"{node_tag}[slot {s['slot']}]",
                content_hash=s["content_hash"],
                get_body_bytes=(lambda vid=s["video"]: _video_bytes_for_upload(vid)),
                file_ext=".mp4",
                content_type="video/mp4",
                asset_type="Video",
                tag=s["tag"],
                group_ref=asset_group,
                project_name=project_name,
                moderation_strategy=moderation,
                force_reupload=force_reupload,
                verify_cached=verify_cached,
                cleanup_staging=cleanup_staging,
                poll_timeout_s=poll_timeout_s,
                ark_access_key=ark_access_key,
                ark_secret_key=ark_secret_key,
                b2_key_id=b2_key_id,
                b2_application_key=b2_application_key,
                b2_bucket=b2_bucket,
                b2_region=b2_region,
            )
            core["slot"] = s["slot"]
            core["elapsed_seconds"] = round(time.time() - slot_t, 3)
            results.append(core)
            if core.get("status") == "failed":
                first_failure_slot = s["slot"]
                first_failure_error = core.get("error") or "register returned status=failed"
                break

        # --- Phase 3: atomic assembly ---
        if first_failure_slot is not None:
            attempted = {r["slot"] for r in results}
            for s in per_slot:
                if s["slot"] not in attempted:
                    results.append({
                        "slot": s["slot"], "content_hash": s["content_hash"], "tag": s["tag"],
                        "status": "unattempted", "asset_url": "", "asset_id": "",
                        "error": f"batch stopped after slot {first_failure_slot} failed",
                    })
            results.sort(key=lambda r: r["slot"])
            overall = (
                f"[{node_tag}] BATCH FAILED at slot {first_failure_slot} of {n}: "
                f"{first_failure_error}. Atomic semantics: emitting empty joined_urls (count=0)."
            )
            print(overall)
            if error_on_fail:
                raise RuntimeError(overall)
            return IO.NodeOutput("", 0, json.dumps(results, indent=2, ensure_ascii=False))

        urls = [r["asset_url"] for r in results]
        joined = "\n".join(urls)
        elapsed = round(time.time() - t_overall, 2)
        print(
            f"[{node_tag}] BATCH OK — {n} slot(s) in {elapsed}s "
            f"({sum(1 for r in results if r['status'] == 'cache_hit')} cache_hit, "
            f"{sum(1 for r in results if r['status'] == 'remote_hit')} remote_hit, "
            f"{sum(1 for r in results if r['status'] == 'uploaded')} uploaded)."
        )
        return IO.NodeOutput(joined, n, json.dumps(results, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Zerogen_ByteplusImageAssetRegister": Zerogen_ByteplusImageAssetRegister,
    "Zerogen_ByteplusVideoAssetRegister": Zerogen_ByteplusVideoAssetRegister,
    "Zerogen_ByteplusImageBatchRegister": Zerogen_ByteplusImageBatchRegister,
    "Zerogen_ByteplusVideoBatchRegister": Zerogen_ByteplusVideoBatchRegister,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Zerogen_ByteplusImageAssetRegister": "BytePlus Image Asset Register",
    "Zerogen_ByteplusVideoAssetRegister": "BytePlus Video Asset Register",
    "Zerogen_ByteplusImageBatchRegister": "BytePlus Image Batch Register",
    "Zerogen_ByteplusVideoBatchRegister": "BytePlus Video Batch Register",
}
