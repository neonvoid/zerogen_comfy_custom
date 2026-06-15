"""Shared utilities for the Seedance 2.0 V2 node pair.

Exports:
  - SEEDANCE_UPLOAD_CONFIG_V2: custom IO type, role-tagged config
  - upload_image_cached / upload_video_cached: MD5-deduped wrappers
  - infer_mode, validate_mode: mode logic for the 3 mutually-exclusive image modes
  - build_config: constructor producing the v2 schema dict
  - mode labels: canonical strings used in config

Design notes:
  - MD5 dedup is process-local. Shared across all nodes in the same ComfyUI
    process. Cache lives for process lifetime; resetting requires restart.
    Enables chunk-chain workflows to reuse an upstream `reference_video` across
    multiple Prep invocations without re-uploading on each chunk.
  - Cache is keyed on raw tensor bytes (via sha256 — hash collisions irrelevant
    for 256-bit space). We call it "MD5 dedup" loosely for brevity.
  - Cache is only for tensor inputs. URL inputs (legacy path on V1) are not
    deduped because they have no tensor to hash.
  - No audio. Deferred until a workflow actually needs audio refs.
"""

from __future__ import annotations

import hashlib
import io
from fractions import Fraction

import torch

from comfy_api.latest import Input, InputImpl
from comfy_api.latest._io import Custom as _IOCustom
from comfy_api.latest._util.video_types import VideoComponents
from comfy_api_nodes.util import (
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
)


# ---------------------------------------------------------------------------
# Custom type
# ---------------------------------------------------------------------------

SEEDANCE_UPLOAD_CONFIG_V2 = _IOCustom("SEEDANCE_UPLOAD_CONFIG_V2")


# ---------------------------------------------------------------------------
# Mode constants (canonical strings — used in config.mode field)
# ---------------------------------------------------------------------------

MODE_TEXT_ONLY = "text_only"
MODE_FIRST_FRAME = "first_frame"      # Mode A: 1 image with role=first_frame
MODE_BRIDGE = "bridge"                 # Mode B: first_frame + last_frame
MODE_MULTIMODAL = "multimodal"         # Mode C: 1-9 images with role=reference_image

VALID_MODES = (MODE_TEXT_ONLY, MODE_FIRST_FRAME, MODE_BRIDGE, MODE_MULTIMODAL)

# Seedance 2.0 multimodal ref mode caps image count at 9.
MAX_MULTIMODAL_IMAGES = 9


# ---------------------------------------------------------------------------
# MD5 dedup cache
# ---------------------------------------------------------------------------

# Process-local cache: { sha256_hex(tensor_bytes) -> uploaded_url }
_UPLOAD_CACHE: dict[str, str] = {}


def _hash_image_tensor(image: torch.Tensor) -> str:
    """Stable content hash for an IMAGE tensor [1, H, W, C] or [H, W, C]."""
    # Contiguous + CPU for reproducible bytes. Don't care about dtype drift
    # between float32/float16 callers — treat as different assets.
    t = image.detach().cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def _hash_video_input(video: Input.Video) -> str:
    """Stable content hash for a VIDEO input.

    Uses the video's frame components if available. Falls back to a
    timestamp-based pseudo-hash (i.e., no dedup) if we can't decode frames —
    better to miss a dedup than to false-match.
    """
    try:
        components = video.get_components()
        frames = components.images  # [F, H, W, C]
        t = frames.detach().cpu().contiguous()
        # Sample frames if video is huge — hashing 200M bytes per call is waste.
        # First frame + last frame + count is a strong-enough identity signal.
        if frames.shape[0] > 2:
            sampled = torch.stack([t[0], t[-1]])
            sig = sampled.numpy().tobytes() + str(frames.shape[0]).encode()
        else:
            sig = t.numpy().tobytes()
        return hashlib.sha256(sig).hexdigest()
    except Exception:
        import time
        return f"nohash_{time.time_ns()}"


# ---------------------------------------------------------------------------
# Cached upload helpers
# ---------------------------------------------------------------------------

# Reverse map: URL → content-hash. Populated alongside the forward _UPLOAD_CACHE
# entries so downstream code (e.g. Moyu asset library dedup) can recover the
# stable content hash from a URL it received via upload_config — important
# because the URL itself rotates per upload (random GCS UUID) while the content
# hash is stable across sessions.
_URL_TO_CONTENT_HASH: dict[str, str] = {}


def get_content_hash_for_url(url: str) -> str | None:
    """Return the content SHA256 (full hex) for a URL previously produced by
    `upload_image_cached` or `upload_video_cached` in this process, or None
    if the URL wasn't seen by these helpers.

    Used by Moyu wrapper's asset library dedup to build content-stable asset
    names independent of the rotating GCS object UUID in the URL itself.
    """
    return _URL_TO_CONTENT_HASH.get(url)


async def upload_image_cached(cls, image: torch.Tensor, wait_label: str = "Uploading image") -> str:
    """Upload an IMAGE tensor to the Comfy asset host, returning the URL.

    Content-addressed: identical tensor bytes → same URL, no re-upload.
    """
    h = _hash_image_tensor(image)
    cached = _UPLOAD_CACHE.get(h)
    if cached is not None:
        _URL_TO_CONTENT_HASH[cached] = h
        print(f"[nv_seedance_upload_utils] upload_image_cached HIT {h[:10]}… → ...{cached[-40:]}")
        return cached

    url = await upload_image_to_comfyapi(cls, image=image, wait_label=wait_label)
    _UPLOAD_CACHE[h] = url
    _URL_TO_CONTENT_HASH[url] = h
    print(f"[nv_seedance_upload_utils] upload_image_cached MISS {h[:10]}… uploaded → ...{url[-40:]}")
    return url


async def upload_video_cached(cls, video: Input.Video, wait_label: str = "Uploading video") -> str:
    """Upload a VIDEO input to the Comfy asset host, returning the URL.

    Dedup uses first-frame + last-frame + frame-count as the content signature.
    Good enough for chain-chunk workflows where the same reference_video is
    wired into multiple Preps.
    """
    h = _hash_video_input(video)
    cached = _UPLOAD_CACHE.get(h)
    if cached is not None:
        _URL_TO_CONTENT_HASH[cached] = h
        print(f"[nv_seedance_upload_utils] upload_video_cached HIT {h[:10]}… → ...{cached[-40:]}")
        return cached

    url = await upload_video_to_comfyapi(cls, video, wait_label=wait_label)
    _UPLOAD_CACHE[h] = url
    _URL_TO_CONTENT_HASH[url] = h
    print(f"[nv_seedance_upload_utils] upload_video_cached MISS {h[:10]}… uploaded → ...{url[-40:]}")
    return url


def encode_frames_to_video(frames: torch.Tensor, fps: float) -> Input.Video:
    """Convenience: wrap an IMAGE [F,H,W,C] tensor + fps into a VIDEO object."""
    return InputImpl.VideoFromComponents(
        VideoComponents(
            images=frames,
            frame_rate=Fraction(int(round(fps))),
        )
    )


def upload_cache_stats() -> dict:
    """For debugging — how many dedup'd entries in the process-local cache."""
    return {"cache_entries": len(_UPLOAD_CACHE)}


# ---------------------------------------------------------------------------
# Mode inference + validation
# ---------------------------------------------------------------------------

def infer_mode(
    has_first_frame: bool,
    has_last_frame: bool,
    has_reference_images: bool,
) -> str:
    """Infer which of the 3 mutually-exclusive image modes is active.

    Returns a canonical MODE_* string. Does NOT validate — use validate_mode
    to check legality of the wiring.
    """
    if has_last_frame:
        return MODE_BRIDGE            # requires first_frame too; validated separately
    if has_first_frame:
        return MODE_FIRST_FRAME
    if has_reference_images:
        return MODE_MULTIMODAL
    return MODE_TEXT_ONLY


def validate_mode(
    *,
    has_first_frame: bool,
    has_last_frame: bool,
    reference_images_count: int,
    has_reference_video: bool,
) -> tuple[bool, str]:
    """Check that the wiring corresponds to one of the 3 legal image modes.

    Per Volcengine docs, the 3 modes are STRICTLY mutually exclusive — Mode A
    (first_frame) and Mode B (first+last_frame) cannot include ANY reference
    media (no reference_video, no reference_audio, no reference_images).
    Multimodal mode (Mode C) is the only mode that combines image + video + audio.

    Returns (ok, message). Used in both Prep VALIDATE_INPUTS and as a safety
    net in the native caller.
    """
    has_ref_imgs = reference_images_count > 0
    in_first_or_last = has_first_frame or has_last_frame

    # Mode A/B (first/last frame) cannot include ANY reference media
    if in_first_or_last and (has_ref_imgs or has_reference_video):
        return False, (
            "Seedance first/last_frame modes (i2v / bridge) CANNOT include reference_video "
            "or reference_images. The 3 image-content modes (first_frame / first+last_frame / "
            "multimodal-reference) are strictly mutually exclusive per Volcengine docs — "
            "multimodal mode is the only mode that combines image + video refs.\n"
            "To use a reference_video alongside an identity anchor image: wire your image to "
            "reference_images (multimodal mode), NOT to first_frame."
        )

    # Mode B requires both frames
    if has_last_frame and not has_first_frame:
        return False, (
            "Bridge mode (first + last frame) requires both first_frame AND last_frame wired. "
            "Either add a first_frame IMAGE or remove last_frame."
        )

    # Multimodal count cap
    if reference_images_count > MAX_MULTIMODAL_IMAGES:
        return False, (
            f"Seedance 2.0 accepts at most {MAX_MULTIMODAL_IMAGES} reference_images "
            f"(got batch size {reference_images_count})."
        )

    # Text-only + video-only + image-only + combined-multimodal are all legal API-wise.
    return True, "ok"


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def build_config(
    *,
    mode: str,
    image_items: list[tuple[str, str]],   # list of (url, role)
    video_url: str | None,
    prompt: str,
    provenance: dict | None = None,
) -> dict:
    """Assemble the v2 SEEDANCE_UPLOAD_CONFIG_V2 dict.

    Validates internally — will raise ValueError on illegal combos so callers
    can fail before emitting a poisoned config.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}. Expected one of {VALID_MODES}.")

    content = []
    for url, role in image_items:
        if role not in ("first_frame", "last_frame", "reference_image"):
            raise ValueError(f"Invalid image role {role!r}.")
        content.append({"kind": "image", "role": role, "url": url})
    if video_url:
        content.append({"kind": "video", "role": "reference_video", "url": video_url})

    n_images = len(image_items)
    n_videos = 1 if video_url else 0

    # Post-hoc validate (belt-and-suspenders)
    ok, msg = validate_mode(
        has_first_frame=any(r == "first_frame" for _, r in image_items),
        has_last_frame=any(r == "last_frame" for _, r in image_items),
        reference_images_count=sum(1 for _, r in image_items if r == "reference_image"),
        has_reference_video=bool(video_url),
    )
    if not ok:
        raise ValueError(f"build_config: mode validation failed — {msg}")

    if mode == MODE_BRIDGE:
        preview_style = "bridge"
    elif mode == MODE_MULTIMODAL:
        preview_style = "grid"
    elif mode == MODE_FIRST_FRAME:
        preview_style = "single"
    else:
        preview_style = "text_only"

    return {
        "schema_version": 2,
        "mode": mode,
        "prompt": prompt,
        "content": content,
        "counts": {"images": n_images, "videos": n_videos},
        "validation": {"ok": True, "message": msg},
        "preview_style": preview_style,
        "provenance": provenance or {},
    }


def extract_content_by_role(config: dict, role: str) -> list[dict]:
    """Pull content items from a v2 config by role (for Native consumer)."""
    return [c for c in config.get("content", []) if c.get("role") == role]
