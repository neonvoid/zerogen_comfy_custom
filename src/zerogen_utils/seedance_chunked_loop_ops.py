"""Pure helpers for NV_SeedanceChunkedLoop.

Split out from seedance_chunked_loop.py so the helpers can be tested in
isolation without the comfy_api / IO.ComfyNode import chain.

Mirrors the kling_chunked_loop_ops.py pattern. Shares 7 model-agnostic
helpers (concurrency, retime, path validation) with kling_chunked_loop_ops
by inline-copy for D-385's runtime-test window — once both nodes are
validated, the shared helpers can be lifted to a `chunked_api_loop_ops.py`
common module.

Seedance-specific helpers:
  - plan_seedance_chunks       — 4-15s integer-duration chunking
  - validate_chunk_seconds     — widget input validation
  - validate_seedance_upload_config_for_chunked — refs from Prep, no ref-video

Shared helpers (inline-copy of kling_chunked_loop_ops):
  - run_chunks_concurrent      — bounded-concurrency asyncio.gather
  - restore_proportional       — proportional select for retime
  - compute_original_from_slowed — reverse the NV_MatchInterpFrames math
  - validate_filename_prefix   — charset guard for chunk filenames
  - validate_output_dir        — absolute-path + write-test
  - preflight_collision_check  — file-exists guard before any API call
  - chunk_filename             — {prefix}_chunkNN.mp4 vs {prefix}.mp4
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Awaitable, Callable, TypeVar

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PREFIX_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
VIDEO_EXT = "mp4"

# Seedance 2.0 hard duration limits (per-call). Validated upstream by
# NV_SeedancePrep for ref-video duration, BUT for the chunked loop the
# constraint applies to each CHUNK's requested output duration.
SEEDANCE_MIN_CHUNK_SECONDS = 4
SEEDANCE_MAX_CHUNK_SECONDS = 15

# Default chunk size — middle of the allowed range, gives 3 chunks per
# ~30s clip and headroom for last-chunk merge if needed.
DEFAULT_TARGET_CHUNK_SECONDS = 10


# ---------------------------------------------------------------------------
# Path / filename validation (inline-copy of kling_chunked_loop_ops helpers)
# ---------------------------------------------------------------------------

def validate_filename_prefix(prefix: str) -> str:
    """Reject paths/spaces/dots. Matches NV_ShotSaverPath custom-name charset."""
    s = (prefix or "").strip()
    if not s:
        raise ValueError("[NV_SeedanceChunkedLoop] filename_prefix is empty.")
    if not VALID_PREFIX_RE.match(s):
        raise ValueError(
            f"[NV_SeedanceChunkedLoop] filename_prefix {s!r} contains disallowed "
            f"characters. Allowed: ^[a-zA-Z0-9_-]+$ (no slashes, spaces, dots)."
        )
    return s


def validate_output_dir(output_dir: str) -> Path:
    """Resolve path, create if missing, write-test. Raises before any API call."""
    s = (output_dir or "").strip()
    if not s:
        raise ValueError(
            "[NV_SeedanceChunkedLoop] output_dir is empty — wire NV_ShotSaverPath "
            "or provide an absolute path."
        )
    p = Path(s).expanduser()
    if not p.is_absolute():
        raise ValueError(
            f"[NV_SeedanceChunkedLoop] output_dir must be absolute, got {s!r}."
        )
    p.mkdir(parents=True, exist_ok=True)
    test = p / f".nv_seedance_chunked_writetest_{os.getpid()}"
    try:
        test.write_text("ok")
        test.unlink()
    except OSError as e:
        raise RuntimeError(
            f"[NV_SeedanceChunkedLoop] output_dir {p} is not writable: "
            f"{e.__class__.__name__}: {e}"
        )
    return p


def chunk_filename(
    prefix: str,
    chunk_index: int,
    ext: str = VIDEO_EXT,
    chunk_count: int | None = None,
) -> str:
    """Filename for a chunk save.

    - When `chunk_count == 1` (single-chunk run): returns `{prefix}.{ext}`
      so downstream loaders that expect a clean filename don't need to know
      about the chunk_NN suffix (drop-in for single-shot).
    - Otherwise: returns `{prefix}_chunk{NN}.{ext}` — NN zero-padded to 2.
    """
    if chunk_count == 1:
        return f"{prefix}.{ext}"
    return f"{prefix}_chunk{chunk_index:02d}.{ext}"


def preflight_collision_check(
    output_dir: Path,
    prefix: str,
    chunk_count: int,
    policy: str,
) -> str:
    """Check for filename collisions BEFORE first API call.

    Returns the effective prefix to use downstream — may differ from input when
    policy=auto_rename and a collision is found.

    Policies:
    - `fail` (safe default): raise FileExistsError if any planned target exists.
    - `overwrite`: silently allow overwrites; returns input prefix unchanged.
    - `auto_rename`: when a collision exists, append `_v2`, `_v3`, ... until an
      uncollided variant is found (cap at 100 attempts). Both old and new files
      are preserved on disk. Returns the modified prefix.
    """
    if policy not in ("fail", "overwrite", "auto_rename"):
        raise ValueError(
            f"on_filename_collision must be 'fail', 'overwrite', or 'auto_rename', got {policy!r}"
        )
    if policy == "overwrite":
        return prefix

    def _collisions_for(p: str) -> list[str]:
        return [
            (output_dir / chunk_filename(p, i, chunk_count=chunk_count)).name
            for i in range(1, chunk_count + 1)
            if (output_dir / chunk_filename(p, i, chunk_count=chunk_count)).exists()
        ]

    head_collisions = _collisions_for(prefix)
    if not head_collisions:
        return prefix

    if policy == "fail":
        raise FileExistsError(
            f"[NV_SeedanceChunkedLoop] {len(head_collisions)} chunk file(s) already "
            f"exist in {output_dir} with policy=fail: {head_collisions[:5]}"
            f"{'...' if len(head_collisions) > 5 else ''}. "
            f"Switch on_filename_collision to 'overwrite' (replace) or 'auto_rename' "
            f"(keep both via {prefix}_vN suffix), or change filename_prefix."
        )

    # policy == "auto_rename" — search for next available _vN suffix
    for n in range(2, 101):
        candidate = f"{prefix}_v{n}"
        if not _collisions_for(candidate):
            print(
                f"[NV_SeedanceChunkedLoop] filename collision auto-rename: "
                f"{prefix!r} → {candidate!r} (existing files preserved)"
            )
            return candidate
    raise RuntimeError(
        f"[NV_SeedanceChunkedLoop] auto_rename exhausted 100 attempts on prefix "
        f"{prefix!r}; output_dir={output_dir} looks saturated. Pick a fresh prefix "
        f"or clean up the directory."
    )


# ---------------------------------------------------------------------------
# Seedance-specific chunk planning
# ---------------------------------------------------------------------------

def validate_chunk_seconds(target_chunk_seconds: int) -> int:
    """Widget input guard: enforce 4-15s integer per Seedance 2.0 API limits."""
    if not isinstance(target_chunk_seconds, int):
        raise ValueError(
            f"[NV_SeedanceChunkedLoop] target_chunk_seconds must be int, got "
            f"{type(target_chunk_seconds).__name__}"
        )
    if target_chunk_seconds < SEEDANCE_MIN_CHUNK_SECONDS or target_chunk_seconds > SEEDANCE_MAX_CHUNK_SECONDS:
        raise ValueError(
            f"[NV_SeedanceChunkedLoop] target_chunk_seconds={target_chunk_seconds} "
            f"out of Seedance allowed range [{SEEDANCE_MIN_CHUNK_SECONDS}, "
            f"{SEEDANCE_MAX_CHUNK_SECONDS}]s."
        )
    return target_chunk_seconds


def plan_seedance_chunks(
    total_frames: int,
    encode_fps: int,
    target_chunk_seconds: int,
) -> list[tuple[int, int]]:
    """Compute (start, end_exclusive) frame ranges for Seedance chunked dispatch.

    Each range corresponds to approximately `target_chunk_seconds` of input
    at `encode_fps`. Frame counts are derived as `round(target_chunk_seconds *
    encode_fps)` for predictable behavior.

    Last-chunk policy:
      - If the final chunk would be smaller than `SEEDANCE_MIN_CHUNK_SECONDS`
        AND merging with the previous chunk would keep the combined size
        ≤ `SEEDANCE_MAX_CHUNK_SECONDS`, MERGE the last chunk into the
        previous one. This avoids the API rejecting a too-short final
        chunk.
      - If the merge would exceed the max, raise with actionable guidance
        (user should reduce target_chunk_seconds).
      - If the WHOLE input is already < 4s, raise — Seedance can't accept
        a sub-4s call.

    Returns a list of (start_idx, end_idx_exclusive) tuples covering [0, total_frames).
    """
    if total_frames < 1:
        raise ValueError(f"total_frames must be >= 1, got {total_frames}")
    if encode_fps < 1:
        raise ValueError(f"encode_fps must be >= 1, got {encode_fps}")
    validate_chunk_seconds(target_chunk_seconds)

    total_seconds = total_frames / encode_fps
    if total_seconds < SEEDANCE_MIN_CHUNK_SECONDS:
        raise ValueError(
            f"[NV_SeedanceChunkedLoop] input is {total_frames} frames "
            f"({total_seconds:.2f}s) at {encode_fps}fps, below Seedance's "
            f"{SEEDANCE_MIN_CHUNK_SECONDS}s per-call minimum. Use the single-"
            f"shot NV_SeedanceRefVideo node for inputs this short."
        )

    chunk_frames = int(round(target_chunk_seconds * encode_fps))
    if chunk_frames < 1:
        raise ValueError(
            f"computed chunk_frames={chunk_frames} from target_chunk_seconds="
            f"{target_chunk_seconds} × encode_fps={encode_fps}"
        )

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_frames, total_frames)
        ranges.append((start, end))
        start = end

    # Merge last chunk if it'd fall below the 4s minimum.
    if len(ranges) >= 2:
        last_start, last_end = ranges[-1]
        last_seconds = (last_end - last_start) / encode_fps
        if last_seconds < SEEDANCE_MIN_CHUNK_SECONDS:
            prev_start, prev_end = ranges[-2]
            combined_seconds = (last_end - prev_start) / encode_fps
            if combined_seconds <= SEEDANCE_MAX_CHUNK_SECONDS:
                ranges[-2] = (prev_start, last_end)
                ranges.pop()
            else:
                # Can't auto-merge — raise with actionable guidance.
                raise ValueError(
                    f"[NV_SeedanceChunkedLoop] last chunk would be "
                    f"{last_seconds:.2f}s (< Seedance {SEEDANCE_MIN_CHUNK_SECONDS}s "
                    f"minimum) and can't merge with previous chunk "
                    f"({combined_seconds:.2f}s > {SEEDANCE_MAX_CHUNK_SECONDS}s "
                    f"maximum). Reduce target_chunk_seconds (try "
                    f"{max(SEEDANCE_MIN_CHUNK_SECONDS, target_chunk_seconds - 2)}) "
                    f"so the last chunk lands above the minimum, or trim/extend "
                    f"input frame count."
                )
    return ranges


def chunk_duration_seconds(
    chunk_start: int,
    chunk_end_exclusive: int,
    encode_fps: int,
) -> int:
    """Per-chunk duration to pass to the Seedance API (integer 4-15s).

    Math: ceil(slice_frames / encode_fps), clamped to [4, 15]. We round UP
    (not nearest) so Seedance is asked for at least as much video as the
    slice represents — any excess frames in the returned MP4 will be
    handled by the global retime restore after concat.
    """
    import math
    slice_frames = chunk_end_exclusive - chunk_start
    if slice_frames < 1:
        raise ValueError(f"empty chunk range [{chunk_start}, {chunk_end_exclusive})")
    raw_seconds = slice_frames / encode_fps
    ceil_seconds = max(SEEDANCE_MIN_CHUNK_SECONDS, int(math.ceil(raw_seconds)))
    return min(ceil_seconds, SEEDANCE_MAX_CHUNK_SECONDS)


# ---------------------------------------------------------------------------
# Slowdown reverse-math + proportional restore
# (inline-copy of kling_chunked_loop_ops helpers — model-agnostic)
# ---------------------------------------------------------------------------

def compute_original_from_slowed(slowed_count: int, slowdown_factor: int) -> int:
    """Reverse NV_MatchInterpFrames math: slowed = (original - 1) * factor + 1.

    factor=1 is a pass-through (no slowdown was applied, restore is a no-op).
    Raises if (slowed - 1) is not divisible by factor.
    """
    if slowdown_factor < 1:
        raise ValueError(
            f"[NV_SeedanceChunkedLoop] slowdown_factor must be >= 1, got {slowdown_factor}"
        )
    if slowed_count < 1:
        raise ValueError(
            f"[NV_SeedanceChunkedLoop] slowed_count must be >= 1, got {slowed_count}"
        )
    if slowdown_factor == 1:
        return slowed_count
    if (slowed_count - 1) % slowdown_factor != 0:
        raise ValueError(
            f"[NV_SeedanceChunkedLoop] input frame count {slowed_count} is not "
            f"consistent with slowdown_factor={slowdown_factor}. Expected "
            f"(slowed - 1) % factor == 0. Did you run NV_MatchInterpFrames "
            f"with interpolation_factor={slowdown_factor} on your raw input? "
            f"If you used a target_frame_count override or a different factor, "
            f"adjust slowdown_factor to match."
        )
    return (slowed_count - 1) // slowdown_factor + 1


def restore_proportional(images: torch.Tensor, original_count: int) -> torch.Tensor:
    """Pick `original_count` evenly-spaced frames from `images`.

    Mirrors NV_RetimeRestore._select_frames. Used for the global retime
    restore after all Seedance chunks have been concatenated.
    """
    available = int(images.shape[0])
    if original_count < 1:
        raise ValueError(f"original_count must be >= 1, got {original_count}")
    if available < 1:
        raise ValueError(f"images must have >= 1 frame, got {available}")
    if original_count == 1:
        return images[0:1]
    step = (available - 1) / (original_count - 1)
    indices = [min(round(i * step), available - 1) for i in range(original_count)]
    return images[indices]


# ---------------------------------------------------------------------------
# Seedance upload-config validation for chunked workflow
# ---------------------------------------------------------------------------

def validate_seedance_upload_config_for_chunked(cfg: dict) -> dict:
    """Validate a SEEDANCE_UPLOAD_CONFIG dict for the chunked workflow.

    The chunked loop uploads its OWN per-chunk reference video, so the
    incoming config must either have NO ref video uploaded (image refs
    only) OR we ignore the ref video and warn.

    Returns the validated config dict.
    """
    if not isinstance(cfg, dict):
        raise ValueError(
            f"[NV_SeedanceChunkedLoop] upload_config must be a dict from "
            f"NV_SeedancePrep, got {type(cfg).__name__}"
        )
    required_keys = ("uploaded_image_urls", "uploaded_video_url", "n_images")
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(
            f"[NV_SeedanceChunkedLoop] upload_config missing keys: {missing}. "
            f"Wire from NV_SeedancePrep's config output."
        )
    image_urls = cfg["uploaded_image_urls"]
    if not isinstance(image_urls, list):
        raise ValueError(
            f"[NV_SeedanceChunkedLoop] upload_config['uploaded_image_urls'] "
            f"must be a list, got {type(image_urls).__name__}"
        )
    if cfg.get("uploaded_video_url") is not None:
        raise ValueError(
            "[NV_SeedanceChunkedLoop] upload_config has a ref video uploaded "
            "(uploaded_video_url is set). The chunked loop uploads its own "
            "per-chunk ref video from the `images` input, so the Prep node's "
            "ref-video role conflicts. DISCONNECT reference_video / "
            "reference_video_frames from NV_SeedancePrep — keep only "
            "reference_images wired to Prep, and wire your long source clip "
            "as `images` here."
        )
    return cfg


def validate_seedance_v2_upload_config_for_chunked(cfg: dict) -> dict:
    """Validate a SEEDANCE_UPLOAD_CONFIG_V2 dict for the native chunked workflow.

    The native chunked loop (NV_SeedanceNativeChunkedLoop_V2) operates
    exclusively in Volcengine's Mode C (multimodal) because that is the
    ONLY mode that allows a reference_video to coexist with reference
    images per the API's mutual-exclusion rule (Volcengine docs:
    "图生视频-首帧、图生视频-首尾帧、多模态参考生视频 ... 互斥场景，不可混用").

    The chunked loop uploads its OWN per-chunk reference video sliced from
    the long source clip, so the incoming config must NOT have a ref
    video already in its content array — it would conflict with the
    per-chunk video upload.

    Returns the validated config dict.
    """
    if not isinstance(cfg, dict):
        raise ValueError(
            f"[NV_SeedanceNativeChunkedLoop] upload_config must be a dict "
            f"from NV_SeedancePrep_V2, got {type(cfg).__name__}"
        )
    required_keys = ("mode", "content")
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(
            f"[NV_SeedanceNativeChunkedLoop] upload_config missing keys: "
            f"{missing}. Wire from NV_SeedancePrep_V2's upload_config output."
        )
    mode = cfg.get("mode")
    if mode != "multimodal":
        raise ValueError(
            f"[NV_SeedanceNativeChunkedLoop] upload_config has mode={mode!r}, "
            f"but the chunked loop requires Mode C (multimodal) — that is the "
            f"only Seedance mode that allows a reference_video alongside the "
            f"ref images. To land in Mode C: wire 1-9 frames to "
            f"NV_SeedancePrep_V2's `reference_images` slot (do NOT wire "
            f"first_frame or last_frame slots — those force Mode A/B which "
            f"forbid reference_video per the API's mutual-exclusion rule)."
        )
    content = cfg.get("content")
    if not isinstance(content, list):
        raise ValueError(
            f"[NV_SeedanceNativeChunkedLoop] upload_config['content'] must "
            f"be a list, got {type(content).__name__}"
        )
    # Refuse configs with a pre-uploaded ref video — the chunked loop
    # uploads its own per-chunk video. Allowing both would either upload
    # two videos per chunk (API likely rejects) or silently drop one.
    pre_uploaded_videos = [c for c in content if isinstance(c, dict) and c.get("kind") == "video"]
    if pre_uploaded_videos:
        raise ValueError(
            f"[NV_SeedanceNativeChunkedLoop] upload_config has "
            f"{len(pre_uploaded_videos)} ref video(s) pre-uploaded in its "
            f"content array. The chunked loop uploads its own per-chunk ref "
            f"video sliced from the `images` input, so the Prep V2 ref-video "
            f"role conflicts. DISCONNECT reference_video / "
            f"reference_video_frames from NV_SeedancePrep_V2 — keep only "
            f"reference_images wired to Prep V2 (lands in Mode C), and wire "
            f"your long source clip as `images` here."
        )
    # Must have at least one image ref (Mode C requires 1-9 images).
    image_refs = [c for c in content if isinstance(c, dict) and c.get("kind") == "image"]
    if not image_refs:
        raise ValueError(
            "[NV_SeedanceNativeChunkedLoop] upload_config is Mode C but has "
            "no reference images in its content array. Mode C requires 1-9 "
            "image refs. Wire at least one frame to NV_SeedancePrep_V2's "
            "reference_images slot."
        )
    # Volcengine API caps ref images at 9. A hand-built config could
    # exceed this; reject loudly with a clear pointer to the limit.
    if len(image_refs) > 9:
        raise ValueError(
            f"[NV_SeedanceNativeChunkedLoop] upload_config has "
            f"{len(image_refs)} reference images. Volcengine Mode C accepts "
            f"a maximum of 9 image refs per request. Reduce the "
            f"reference_images batch size wired to NV_SeedancePrep_V2."
        )
    # Each image item must have a usable string URL and the right Mode C
    # role. Hand-built configs that copy Mode A's `first_frame` role into
    # a "mode=multimodal" wrapper would pass the bare mode check but the
    # API would reject mid-flight; catch at graph-build time instead.
    for i, item in enumerate(image_refs):
        url = item.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(
                f"[NV_SeedanceNativeChunkedLoop] upload_config image ref "
                f"#{i} has missing or empty 'url' (got {url!r}). Wire valid "
                f"ref images through NV_SeedancePrep_V2 — do not hand-build "
                f"the config."
            )
        role = item.get("role")
        if role != "reference_image":
            raise ValueError(
                f"[NV_SeedanceNativeChunkedLoop] upload_config image ref "
                f"#{i} has role={role!r}, but Mode C requires "
                f"role='reference_image' for every image. Roles 'first_frame' "
                f"/ 'last_frame' belong to Mode A/B and are forbidden in "
                f"Mode C per Volcengine's mutual-exclusion rule. Wire refs "
                f"only via NV_SeedancePrep_V2.reference_images slot."
            )
    return cfg


# ---------------------------------------------------------------------------
# Bounded-concurrency runner for parallel chunk dispatch
# (inline-copy of kling_chunked_loop_ops.run_chunks_concurrent — identical logic)
# ---------------------------------------------------------------------------

T = TypeVar("T")


async def run_chunks_concurrent(
    chunk_factories: list[Callable[[], Awaitable[T]]],
    max_concurrent: int,
) -> list[T | BaseException]:
    """Run a list of async chunk callables with bounded concurrency.

    Each callable in ``chunk_factories`` is a zero-arg factory that returns
    an awaitable (e.g. ``lambda: process_chunk(i)``). At most
    ``max_concurrent`` coroutines run inside the semaphore-gated body at
    any one time. Results returned in INPUT ORDER, with exceptions
    captured in-place via ``asyncio.gather(return_exceptions=True)``.
    """
    if not isinstance(max_concurrent, int) or max_concurrent < 1:
        raise ValueError(
            f"max_concurrent must be int >= 1, got {max_concurrent!r}"
        )
    if not chunk_factories:
        return []

    sem = asyncio.Semaphore(max_concurrent)

    async def _gated(factory: Callable[[], Awaitable[T]]) -> T:
        async with sem:
            return await factory()

    return await asyncio.gather(
        *(_gated(f) for f in chunk_factories),
        return_exceptions=True,
    )
