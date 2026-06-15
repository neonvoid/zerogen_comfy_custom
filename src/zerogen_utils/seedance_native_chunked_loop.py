"""NV Seedance Native Chunked Loop V2 — multi-chunk Seedance 2.0 via native ark- key.

Native counterpart of NV_SeedanceChunkedLoop (D-386, proxy fork). Routes
through Volcengine's Ark endpoint directly using the user's ark- key
(VOLCENGINE_ARK_API_KEY / ARK_API_KEY / .env file), matching the
NV_SeedanceNativeRefVideo_V2 single-shot path that is the user's
runtime-validated production node (2026-04-23, $1.33/5s Pro 720p).

Mirrors the NV_KlingChunkedLoop architecture exactly:

  1. Slice the long source clip into 4-15s chunks at encode_fps.
  2. For each chunk: encode + upload as Seedance ref video, submit task
     to Volcengine, poll with transient-failure backoff, download, save
     to disk, decode frames.
  3. Concatenate outputs in chunk order.
  4. Optional retime restore via slowdown_factor.
  5. Optional parallel dispatch via max_concurrent.

Volcengine API constraints baked into this node:

  - Mode C (multimodal) ONLY. Per Volcengine docs the 3 image-content
    modes are mutually exclusive — Mode A (first_frame) and Mode B
    (first+last_frame) CANNOT include a reference_video. The chunked
    loop's per-chunk ref-video pattern only fits Mode C.
    `validate_seedance_v2_upload_config_for_chunked` enforces this at
    graph-build time.

  - Per-call duration is 4-15s integer.

  - Long jobs: 5s Pro 720p ≈ 5 min wall, 15s Pro Mode C with 5 refs +
    ref_video can be ~75 min (memory: seedance_fork.md, measured
    2026-04-24). Per-chunk poll timeout default = 1800s (30 min).

  - Transient network failures during poll are retried via the V2
    native's `_poll_task` helper (10 consecutive failures = real outage).
    A single TLS reset or DNS hiccup must NOT forfeit the whole job;
    Volcengine's server-side task continues regardless of polling
    client state.

  - Per-chunk task IDs are surfaced as a top-level JSON STRING output
    so the user can recover hung chunks via NV_SeedanceFetchTask if the
    node itself times out or is interrupted.

Reuses helpers from seedance_chunked_loop_ops.py — chunk planning,
retime restore, parallel-dispatch runner, path/filename validators are
all model-agnostic (D-386 and this node share the same scaffolding).
Only the per-chunk API-call body differs.

Reuses helpers from nv_seedance_native_v2.py (private underscore
names) — `_post_task`, `_get_task`, `_poll_task`, `_classify_http_error`,
`_token_usage_summary`, `_API_BASE`, `_CREATE_PATH`, `_STATUS_PATH`,
`_MODELS`, `_PRICE_PER_1K_TOKENS`. The underscore prefix is a soft
"sibling-internal" convention; importing across sibling files in the
same package is intentional here to avoid drift between single-shot
and chunked variants. Plan: lift these to a shared
`nv_seedance_native_api_helpers.py` module after this node validates,
so both variants explicitly share contract.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Callable

import aiohttp
import torch
import torch.nn.functional as F

from comfy_api.latest import IO, Input, InputImpl
from comfy_api.latest._util.video_types import VideoComponents
from comfy_api_nodes.util import download_url_to_video_output

from .api_keys import resolve_api_key
from .nv_seedance_native_v2 import (
    _API_BASE,
    _CREATE_PATH,
    _MODELS,
    _PRICE_PER_1K_TOKENS,
    _STATUS_PATH,
    _auto_inject_tags,
    _classify_http_error,
    _get_task,
    _poll_task,
    _post_task,
    _token_usage_summary,
)
from .nv_seedance_upload_utils import (
    MODE_MULTIMODAL,
    SEEDANCE_UPLOAD_CONFIG_V2,
    upload_video_cached,
)
from .seedance_chunked_loop_ops import (
    DEFAULT_TARGET_CHUNK_SECONDS as _DEFAULT_TARGET_CHUNK_SECONDS,
    SEEDANCE_MAX_CHUNK_SECONDS as _SEEDANCE_MAX_CHUNK_SECONDS,
    SEEDANCE_MIN_CHUNK_SECONDS as _SEEDANCE_MIN_CHUNK_SECONDS,
    chunk_duration_seconds as _chunk_duration_seconds,
    chunk_filename as _chunk_filename,
    compute_original_from_slowed as _compute_original_from_slowed,
    plan_seedance_chunks as _plan_seedance_chunks,
    preflight_collision_check as _preflight_collision_check,
    restore_proportional as _restore_proportional,
    run_chunks_concurrent as _run_chunks_concurrent,
    validate_filename_prefix as _validate_filename_prefix,
    validate_output_dir as _validate_output_dir,
    validate_seedance_v2_upload_config_for_chunked as _validate_upload_config_v2,
)

try:
    from comfy.model_management import (
        InterruptProcessingException as _InterruptProcessingException,
        throw_exception_if_processing_interrupted as _check_interrupt,
    )
except ImportError:
    class _InterruptProcessingException(Exception):  # type: ignore[no-redef]
        """Fallback stub for dev environments without comfy installed."""
        pass

    def _check_interrupt() -> None:
        pass


# Volcengine ref-video pixel budget (matches Prep V2's constraint).
_REF_VIDEO_PIXEL_MIN = 409_600
_REF_VIDEO_PIXEL_MAX = 2_086_876

# Resolution + ratio options (match native V2).
_RESOLUTIONS = ["480p", "720p", "1080p"]
_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]


# ---------------------------------------------------------------------------
# Deferred .tmp cleanup (same pattern as D-386, D-385 follow-up)
# ---------------------------------------------------------------------------

def _write_in_flight_recovery_sidecar(
    out_dir: Path,
    prefix: str,
    in_flight: dict,
    reason: str,
) -> None:
    """Persist in-flight task IDs to disk for recovery via NV_SeedanceFetchTask.

    Called BEFORE raising InterruptProcessingException (cancel) or
    RuntimeError (chunk failure with siblings in-flight) so the user has
    a manifest of submitted-but-unresolved Volcengine tasks even if the
    node never returns through the normal path. Sidecar file:
    `{out_dir}/{prefix}_in_flight_task_ids.json`.

    `in_flight` is a dict {chunk_idx: task_id} of tasks that have been
    submitted to Volcengine but haven't successfully completed locally.
    Volcengine has no remote cancel, so these tasks will continue
    server-side regardless of our cancellation.
    """
    if not in_flight:
        return
    try:
        # Re-assert dir exists — same SMB-idle-disconnect defense as
        # _download_url_to_file. Without this, a dropped mapped drive
        # silently loses the task_id (real-money recovery info).
        out_dir.mkdir(parents=True, exist_ok=True)
        sidecar = out_dir / f"{prefix}_in_flight_task_ids.json"
        sidecar.write_text(
            json.dumps({
                "reason": reason,
                "timestamp": time.time(),
                "in_flight_task_ids_by_chunk_idx": {
                    str(k): v for k, v in in_flight.items()
                },
                "note": (
                    "These Volcengine tasks were submitted before the chunked "
                    "loop was interrupted or failed. They continue server-side. "
                    "Recover each output via NV_SeedanceFetchTask with the "
                    "task_id."
                ),
            }, indent=2),
            encoding="utf-8",
        )
        print(
            f"[NV_SeedanceNativeChunkedLoop] Wrote in-flight task_id "
            f"recovery sidecar ({len(in_flight)} task(s)) → {sidecar}"
        )
    except OSError as e:
        # Sidecar write failure must NOT prevent the interrupt from
        # propagating — just log so the user has the task_ids in console.
        print(
            f"[NV_SeedanceNativeChunkedLoop] WARN: couldn't write recovery "
            f"sidecar ({e.__class__.__name__}: {e}). In-flight task_ids "
            f"(chunk_idx → task_id): {dict(in_flight)}"
        )


def _deferred_unlink_retry(path: Path, max_attempts: int = 60, delay_sec: float = 0.5) -> None:
    """Best-effort deferred cleanup of a tmp file the current thread can't unlink.

    See seedance_chunked_loop.py for the full rationale — same pattern
    handles the Windows asyncio.to_thread cancellation file-lock race
    where the worker may still hold the .tmp open when finally-block
    unlink fires.
    """
    import threading

    def _retry() -> None:
        import time as _time
        for _ in range(max_attempts):
            _time.sleep(delay_sec)
            try:
                path.unlink(missing_ok=True)
                return
            except OSError:
                continue

    threading.Thread(target=_retry, daemon=True).start()


async def _download_url_to_file(url: str, target_path: Path) -> int:
    """Stream URL to {target}.tmp, then os.replace atomically. Returns bytes written.

    Best-effort cleanup: same contract as D-386's helper — .tmp is removed
    on every exit path including Windows file-lock-after-cancel via daemon-
    thread retry (30s window). Orphans persist as harmless .tmp files in
    output_dir if the worker download exceeds 30s after cancellation.

    Re-asserts the target directory exists before writing — defends against
    Windows SMB idle-disconnect dropping a mapped drive during a long
    Seedance gen (5-75 min). validate_output_dir mkdir at execute() start
    can be stale by the time the download fires.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")

    def _do_download() -> int:
        with urllib.request.urlopen(url) as r, open(tmp_path, "wb") as out:
            total = 0
            while True:
                chunk = r.read(64 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
        return total

    try:
        written = await asyncio.to_thread(_do_download)
        os.replace(tmp_path, target_path)
        return written
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                _deferred_unlink_retry(tmp_path)


# ---------------------------------------------------------------------------
# Pixel-budget guard (one-shot upfront — H,W constant across chunks)
# ---------------------------------------------------------------------------

def _even_dims(h: int, w: int) -> tuple[int, int]:
    return h + (h % 2), w + (w % 2)


def _clamp_frames_to_pixel_budget(frames: torch.Tensor) -> tuple[torch.Tensor, bool, tuple[int, int]]:
    """Bilinear-downscale frames [N,H,W,C] so pixels-per-frame fit Volcengine budget.

    Mirrors nv_seedance_prep_v2._clamp_video_frames_to_budget for the chunked
    workflow. Returns (frames, did_resize, (new_h, new_w)). Min-pixel
    violation raises (no auto-upscale — would add blur).
    """
    h, w = frames.shape[1], frames.shape[2]
    pixels = h * w

    if pixels > _REF_VIDEO_PIXEL_MAX:
        scale = (_REF_VIDEO_PIXEL_MAX / pixels) ** 0.5
        new_h = max(2, int(h * scale))
        new_w = max(2, int(w * scale))
        new_h, new_w = _even_dims(new_h, new_w)
        x = frames.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1), True, (new_h, new_w)

    if pixels < _REF_VIDEO_PIXEL_MIN:
        raise ValueError(
            f"[NV_SeedanceNativeChunkedLoop] input is {w}x{h} = {pixels:,}px, under "
            f"Volcengine's {_REF_VIDEO_PIXEL_MIN:,}px minimum (~{int(_REF_VIDEO_PIXEL_MIN ** 0.5)}x"
            f"{int(_REF_VIDEO_PIXEL_MIN ** 0.5)}). Upscale upstream — auto-upscale not "
            f"supported (would add blur)."
        )

    new_h, new_w = _even_dims(h, w)
    if (new_h, new_w) != (h, w):
        x = frames.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1), True, (new_h, new_w)

    return frames, False, (h, w)


# ---------------------------------------------------------------------------
# Per-chunk API call (Volcengine native endpoint)
# ---------------------------------------------------------------------------

async def _seedance_native_chunk_api_call(
    cls,
    chunk_images: torch.Tensor,
    encode_fps: int,
    final_prompt: str,
    image_content_items: list[dict],
    model_id: str,
    resolution: str,
    ratio: str,
    duration_s: int,
    generate_audio: bool,
    watermark: bool,
    seed: int,
    return_last_frame: bool,
    api_key: str,
    poll_interval_s: float,
    poll_timeout_s: float,
    chunk_label: str,
    dump_request_to: Path | None = None,
    on_task_submitted: Callable[[str], None] | None = None,
) -> tuple[torch.Tensor, dict, float, dict, str]:
    """Encode + upload + submit + poll + download for ONE Seedance native chunk.

    Returns (output_images, chunk_meta, output_fps, token_usage_dict, task_id).
    Raises on errors (including transient outage after 10 consecutive failures
    — `task_id` will be in the error message for NV_SeedanceFetchTask recovery).
    """
    t_start = time.time()
    num_frames = chunk_images.shape[0]
    h, w = chunk_images.shape[1], chunk_images.shape[2]

    print(
        f"[NV_SeedanceNativeChunkedLoop {chunk_label}] Encoding {num_frames} frames at "
        f"{encode_fps}fps, {w}x{h}, duration={duration_s}s, model={model_id}, "
        f"res={resolution} ratio={ratio}"
    )

    # Encode chunk frames → VIDEO → upload to Volcengine (cached by MD5)
    chunk_video = InputImpl.VideoFromComponents(
        VideoComponents(images=chunk_images, frame_rate=Fraction(encode_fps))
    )
    chunk_video_url = await upload_video_cached(
        cls, chunk_video, wait_label=f"Uploading chunk ref video ({chunk_label})"
    )

    # Build content array — shared image refs from config + this chunk's video
    content: list[dict] = [{"type": "text", "text": final_prompt}]
    for item in image_content_items:
        content.append({
            "type": "image_url",
            "image_url": {"url": item["url"]},
            "role": item.get("role", "reference_image"),
        })
    content.append({
        "type": "video_url",
        "video_url": {"url": chunk_video_url},
        "role": "reference_video",
    })

    # Build Volcengine task creation payload (matches native V2's shape)
    payload: dict = {
        "model": model_id,
        "content": content,
        "generate_audio": generate_audio,
        "resolution": resolution,
        "ratio": ratio,
        "duration": duration_s,
        "watermark": watermark,
        "return_last_frame": return_last_frame,
    }
    if seed != -1:
        payload["seed"] = seed

    if dump_request_to is not None:
        try:
            # Trim image_url / video_url to last 40 chars in the dump for readability;
            # the actual payload sent to the API has the full URLs.
            dump_content = []
            for item in content:
                d = dict(item)
                if "image_url" in d and isinstance(d["image_url"], dict):
                    u = d["image_url"].get("url", "")
                    d["image_url"] = {"url_tail": u[-40:] if u else None}
                if "video_url" in d and isinstance(d["video_url"], dict):
                    u = d["video_url"].get("url", "")
                    d["video_url"] = {"url_tail": u[-40:] if u else None}
                dump_content.append(d)
            dump_payload = dict(payload)
            dump_payload["content"] = dump_content
            dump_payload["_meta"] = {
                "endpoint": f"{_API_BASE}{_CREATE_PATH}",
                "method": "POST",
                "chunk_label": chunk_label,
                "encode_fps": encode_fps,
                "ref_image_count": len(image_content_items),
            }
            dump_request_to.parent.mkdir(parents=True, exist_ok=True)
            dump_request_to.write_text(
                json.dumps(dump_payload, indent=2, default=str), encoding="utf-8"
            )
            print(f"[NV_SeedanceNativeChunkedLoop {chunk_label}] dump_request → {dump_request_to}")
        except OSError as _e:
            print(f"[NV_SeedanceNativeChunkedLoop {chunk_label}] WARNING: dump_request write failed: {_e}")

    t_submit = time.time()

    # Each chunk owns its own aiohttp session — matches native V2's per-call pattern.
    session_timeout = aiohttp.ClientTimeout(
        total=None, connect=30, sock_connect=30, sock_read=120,
    )
    connector = aiohttp.TCPConnector(force_close=True, limit=8)
    session = aiohttp.ClientSession(timeout=session_timeout, connector=connector)
    task_id = ""
    try:
        create_resp = await _post_task(session, api_key, payload)
        task_id = create_resp.get("id", "")
        if not task_id:
            raise RuntimeError(
                f"[NV_SeedanceNativeChunkedLoop {chunk_label}] Task creation "
                f"returned no id. Raw: {create_resp}"
            )
        # Surface task_id to the caller BEFORE polling — lets the outer
        # dispatch record it for interrupt-recovery sidecars even if poll
        # raises InterruptProcessingException mid-flight.
        if on_task_submitted is not None:
            try:
                on_task_submitted(task_id)
            except Exception as cb_e:
                # Callback failure must NOT prevent the task from polling
                # to completion; just log and continue.
                print(
                    f"[NV_SeedanceNativeChunkedLoop {chunk_label}] WARN: "
                    f"on_task_submitted callback raised "
                    f"{cb_e.__class__.__name__}: {cb_e}"
                )
        print(f"[NV_SeedanceNativeChunkedLoop {chunk_label}] Task submitted: {task_id}")
        final_resp = await _poll_task(
            session, api_key, task_id, poll_interval_s, poll_timeout_s
        )
    finally:
        await session.close()
        await asyncio.sleep(0.1)  # Windows ProactorEventLoop SSL cleanup tick

    t_done = time.time()

    status = final_resp.get("status")
    if status != "succeeded":
        err = final_resp.get("error") or {}
        raise RuntimeError(
            f"[NV_SeedanceNativeChunkedLoop {chunk_label}] task ended status="
            f"{status}. error.code={err.get('code')!r} "
            f"error.message={err.get('message')!r}. task_id={task_id}"
        )

    resp_content = final_resp.get("content") or {}
    result_video_url = resp_content.get("video_url")
    if not result_video_url:
        raise RuntimeError(
            f"[NV_SeedanceNativeChunkedLoop {chunk_label}] task succeeded but "
            f"content.video_url missing. task_id={task_id}. Raw: {final_resp}"
        )

    # Decode frames (used for tensor output; the MP4 is also saved to disk separately).
    output_video = await download_url_to_video_output(result_video_url)
    try:
        components = output_video.get_components()
        output_images = components.images
        output_fps = float(components.frame_rate)
        output_frames = int(output_images.shape[0])
    except Exception as e:
        print(f"[NV_SeedanceNativeChunkedLoop {chunk_label}] WARN: frame decode failed: {e}")
        output_images = torch.zeros(1, 64, 64, 3)
        output_fps = 0.0
        output_frames = 0

    t_end = time.time()

    token_usage = _token_usage_summary(final_resp, model_id, has_video=True)

    chunk_meta = {
        "task_id": task_id,
        "result_video_url": result_video_url,
        "duration_s": duration_s,
        "input_frames": num_frames,
        "input_fps": encode_fps,
        "input_resolution": f"{w}x{h}",
        "output_frames": output_frames,
        "output_fps": output_fps,
        "output_resolution": f"{output_images.shape[2]}x{output_images.shape[1]}",
        "status": status,
        "model_echo": final_resp.get("model"),
        "ratio_echo": final_resp.get("ratio"),
        "resolution_echo": final_resp.get("resolution"),
        "duration_echo": final_resp.get("duration"),
        "token_usage": token_usage,
        "timing": {
            "upload_sec": round(t_submit - t_start, 1),
            "api_processing_sec": round(t_done - t_submit, 1),
            "download_sec": round(t_end - t_done, 1),
            "total_sec": round(t_end - t_start, 1),
        },
    }
    return output_images, chunk_meta, output_fps, token_usage, task_id


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class NV_SeedanceNativeChunkedLoop_V2(IO.ComfyNode):
    """Loop Seedance 2.0 native ref-to-video across N chunks (Mode C only).

    Native counterpart of NV_SeedanceChunkedLoop (proxy). Uses your ark-
    key from VOLCENGINE_ARK_API_KEY / ARK_API_KEY env var or .env file.
    For inputs that fit in a single 4-15s call, NV_SeedanceNativeRefVideo_V2
    single-shot stays the right node.

    Mode C (multimodal) is the ONLY supported mode — per Volcengine API,
    Mode A (first_frame) and Mode B (first+last_frame) forbid
    reference_video, which the chunked loop must include per chunk.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_SeedanceNativeChunkedLoop_V2",
            display_name="NV Seedance Native Chunked Loop V2",
            category="NV_Utils/api",
            description=(
                "Loop Seedance 2.0 native ref-to-video across N chunks of a "
                "long source clip in a single queue run. Mode C (multimodal) "
                "only — chunked dispatch uploads a per-chunk ref video which "
                "is only allowed in Mode C. Wire SEEDANCE_UPLOAD_CONFIG_V2 "
                "from NV_SeedancePrep_V2 with ref images (no ref video). "
                "Saves raw Seedance output per chunk to disk. Set "
                "slowdown_factor>1 to restore the joined output back to "
                "your raw pre-NV_MatchInterpFrames frame count. Set "
                "max_concurrent>1 to fire chunks in parallel — particularly "
                "impactful since 15s Mode C Pro jobs can be ~75 min wall."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    force_input=True,
                    tooltip=(
                        "Final prompt sent to Seedance for EVERY chunk. "
                        "Typically wired from NV_PromptRefiner (mode="
                        "seedance_ref). v1 uses the same prompt for all "
                        "chunks. CN + [图1]/[视频1] bracket tags beat EN "
                        "+ @Image1/@Video1 per Seedance training data."
                    ),
                ),
                IO.Image.Input(
                    "images",
                    tooltip=(
                        "Pre-prepared source frames [B,H,W,C]. For slowdown "
                        "workflow, run NV_MatchInterpFrames(interpolation_"
                        "factor=N) UPSTREAM and pass the slowed tensor here, "
                        "then set slowdown_factor=N below for global restore. "
                        "Each chunk's slice is uploaded as the per-chunk "
                        "Seedance reference video."
                    ),
                ),
                SEEDANCE_UPLOAD_CONFIG_V2.Input(
                    "upload_config",
                    tooltip=(
                        "SEEDANCE_UPLOAD_CONFIG_V2 from NV_SeedancePrep_V2. "
                        "Provides the SHARED reference IMAGES (uploaded once, "
                        "reused per chunk). MUST be Mode C (multimodal) — "
                        "wire 1-9 frames to Prep V2's `reference_images` "
                        "slot. DO NOT wire first_frame / last_frame / "
                        "reference_video on Prep V2 — those force Mode A/B "
                        "which forbid reference_video per Volcengine's API. "
                        "The chunked loop uploads its own per-chunk ref "
                        "video from the `images` input here."
                    ),
                ),
                IO.String.Input(
                    "output_dir",
                    force_input=True,
                    tooltip=(
                        "Absolute path to directory for per-chunk save. "
                        "Wire from NV_ShotSaverPath.output_dir."
                    ),
                ),
                IO.String.Input(
                    "filename_prefix",
                    default="seedance_native_chunked",
                    tooltip=(
                        "Filename prefix for saved chunks. Multi-chunk: "
                        "{prefix}_chunk{NN}.mp4. Single-chunk: {prefix}.mp4. "
                        "Charset ^[a-zA-Z0-9_-]+$."
                    ),
                ),
                IO.Combo.Input(
                    "on_filename_collision",
                    options=["fail", "overwrite", "auto_rename"],
                    default="fail",
                    tooltip=(
                        "Behavior when a target chunk file already exists in "
                        "output_dir:\n"
                        "  • fail (default): raises BEFORE any API call — "
                        "    safest, prevents accidental data loss\n"
                        "  • overwrite: silently replaces existing files — "
                        "    use when you intentionally want fresh output\n"
                        "  • auto_rename: appends _v2/_v3/... to the prefix "
                        "    until an uncollided variant is found (cap 100). "
                        "    Both old and new files are preserved on disk. "
                        "    Useful for iteration without losing prior takes."
                    ),
                ),
                IO.Boolean.Input(
                    "dump_request",
                    default=False,
                    tooltip=(
                        "Diagnostic: when True, writes the outgoing Seedance "
                        "request payload to disk as JSON BEFORE the API call, "
                        "at {output_dir}/{prefix}_chunkNN_request.json. URLs "
                        "are truncated to last 40 chars for readability. "
                        "Works in both live and debug modes."
                    ),
                ),
                IO.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip=(
                        "DRY RUN — no API calls, no uploads, no per-chunk "
                        "MP4 saves, no charges. (dump_request JSON files "
                        "DO still write when enabled — that's the whole "
                        "point of the dump flag.) Runs every validation + "
                        "planning step, prints chunk count + plan, then "
                        "uses INPUT chunk frames as mock Seedance output. "
                        "Concat + retime restore still run."
                    ),
                ),
                IO.Int.Input(
                    "fps",
                    default=24,
                    min=1,
                    max=60,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=(
                        "Source/encode FPS. Used to plan chunk boundaries "
                        "(chunk_frames = target_chunk_seconds × fps) and "
                        "to encode the per-chunk ref video upload."
                    ),
                ),
                IO.Int.Input(
                    "target_chunk_seconds",
                    default=_DEFAULT_TARGET_CHUNK_SECONDS,
                    min=_SEEDANCE_MIN_CHUNK_SECONDS,
                    max=_SEEDANCE_MAX_CHUNK_SECONDS,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip=(
                        f"Target seconds per chunk ({_SEEDANCE_MIN_CHUNK_SECONDS}-"
                        f"{_SEEDANCE_MAX_CHUNK_SECONDS}s integer per Seedance API). "
                        "Default 10s balances cost-per-call and chunk count. "
                        "Last chunk auto-merges with previous if it'd fall "
                        "below 4s; raises if merge would exceed 15s (reduce "
                        "this value if it raises)."
                    ),
                ),
                IO.Combo.Input(
                    "model",
                    options=list(_MODELS.keys()),
                    default="Seedance 2.0 Pro",
                    tooltip=(
                        "Pro = quality, Fast = ~20% cheaper. Same model used "
                        "for every chunk. Pro supports 1080p; Fast does not."
                    ),
                ),
                IO.Combo.Input(
                    "resolution",
                    options=_RESOLUTIONS,
                    default="720p",
                    tooltip="480p / 720p / 1080p (1080p Pro-only).",
                ),
                IO.Combo.Input(
                    "ratio",
                    options=_RATIOS,
                    default="adaptive",
                    tooltip=(
                        "Output aspect ratio. 'adaptive' lets the model pick. "
                        "Same ratio for every chunk."
                    ),
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    tooltip=(
                        "Produce synchronized audio per chunk. Default OFF "
                        "for chunked workflows: Gemini hypothesis (per "
                        "seedance_fork.md) is that audio coupling warps "
                        "face muscles to match lip-sync phonemes, "
                        "degrading photoreal identity retention. For "
                        "face-swap shots disable; for non-face content "
                        "can enable."
                    ),
                ),
                IO.Int.Input(
                    "slowdown_factor",
                    default=1,
                    min=1,
                    max=8,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=(
                        "If you ran NV_MatchInterpFrames(interpolation_factor=N) "
                        "upstream, set this to N. After concat, the joined "
                        "output is restored to the raw pre-slowdown frame "
                        "count via proportional select. 1 = no slowdown "
                        "applied (passthrough — joined_images = raw concat). "
                        "Restore math: original_frames = (input_frames - 1) / "
                        "factor + 1. Raises if input frame count is "
                        "inconsistent with the factor."
                    ),
                ),
                IO.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="-1 = random per chunk. Non-deterministic even at fixed seed.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Add Volcengine watermark to each chunk output.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "return_last_frame",
                    default=False,
                    tooltip=(
                        "Request last_frame PNG per chunk. Chunked loop "
                        "doesn't currently chain last_frame across chunks "
                        "(parallel mode has no chronological ordering); "
                        "leave OFF unless you want the last frame URLs "
                        "in per-chunk metadata for downstream wiring."
                    ),
                    optional=True,
                ),
                IO.Float.Input(
                    "poll_interval_s",
                    default=9.0,
                    min=2.0,
                    max=60.0,
                    step=1.0,
                    tooltip="Seconds between status polls per chunk.",
                    optional=True,
                ),
                IO.Float.Input(
                    "poll_timeout_s",
                    default=5400.0,
                    min=60.0,
                    max=7200.0,
                    step=60.0,
                    tooltip=(
                        "Max seconds to wait for a single chunk's task to "
                        "complete. Default 5400s (1.5 hr) — sized to "
                        "outlast Mode C Pro 15s with 5 refs + ref_video "
                        "which can take ~75 min in the upper tail (per "
                        "seedance_fork.md, measured 2026-04-24). Most "
                        "chunks finish in 5-10 min; the long ceiling only "
                        "matters for outliers. Raising to 7200s (2 hr) is "
                        "safe — Volcengine bills per-token regardless of "
                        "local poll timeout. On local timeout, task_id is "
                        "logged in per_chunk_metadata AND the top-level "
                        "task_ids output AND the in-flight recovery "
                        "sidecar — recover via NV_SeedanceFetchTask."
                    ),
                    optional=True,
                ),
                IO.String.Input(
                    "api_key",
                    default="",
                    tooltip="Optional ark- key override. Empty → env / .env.",
                    optional=True,
                ),
                IO.Int.Input(
                    "max_concurrent",
                    default=1,
                    min=1,
                    max=8,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=(
                        "Maximum number of Seedance API chunks in flight at "
                        "once. 1 = sequential (default — preserves fail-and-"
                        "stop semantics on first error). >1 = parallel via "
                        "asyncio.gather with a semaphore-bound concurrency "
                        "cap. Particularly impactful for Seedance native, "
                        "where per-call latency can be 5-75 min — a 4-chunk "
                        "parallel run completes in ~max(per-chunk) wall "
                        "time vs ~sum(per-chunk) sequential.\n\n"
                        "Parallel mode trade-offs:\n"
                        "  • Cost: token-based, same per-chunk billing, no "
                        "concurrency discount.\n"
                        "  • Rate limits: Volcengine per-account caps are "
                        "undocumented; start at 2-3 and raise if no 429s.\n"
                        "  • Failure semantics: in-flight chunks are NON-"
                        "CANCELLABLE (Volcengine has no remote cancel — "
                        "the server-side task runs to completion regardless "
                        "of our polling). All dispatched chunks complete; "
                        "you may be billed for chunks that ran after the "
                        "first failure. Already-saved chunks stay on disk; "
                        "the node raises with a summary at the end.\n"
                        "  • Cancel button: works between chunks the "
                        "semaphore hasn't released yet, but cannot stop "
                        "chunks already polling Volcengine.\n"
                        "  • Per-chunk task_ids are surfaced as a JSON "
                        "STRING output so you can recover any chunk via "
                        "NV_SeedanceFetchTask if poll times out."
                    ),
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="joined_images"),
                IO.Float.Output(display_name="output_fps"),
                IO.Int.Output(display_name="output_frames"),
                IO.String.Output(display_name="final_prompt"),
                IO.String.Output(display_name="task_ids"),
                IO.String.Output(display_name="per_chunk_metadata"),
                IO.Float.Output(display_name="total_estimated_usd"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        images: Input.Image,
        upload_config: dict,
        output_dir: str,
        filename_prefix: str,
        on_filename_collision: str,
        dump_request: bool,
        debug_mode: bool,
        fps: int,
        target_chunk_seconds: int,
        model: str,
        resolution: str,
        ratio: str,
        generate_audio: bool,
        slowdown_factor: int = 1,
        seed: int = -1,
        watermark: bool = False,
        return_last_frame: bool = False,
        poll_interval_s: float = 9.0,
        poll_timeout_s: float = 5400.0,
        api_key: str = "",
        max_concurrent: int = 1,
    ) -> IO.NodeOutput:
        t_overall_start = time.time()

        # ---------- Phase 0: validation BEFORE any API call ----------
        if not isinstance(max_concurrent, int) or max_concurrent < 1 or max_concurrent > 8:
            raise ValueError(
                f"[NV_SeedanceNativeChunkedLoop] max_concurrent must be int "
                f"in [1, 8], got {max_concurrent!r}"
            )

        prefix = _validate_filename_prefix(filename_prefix)
        out_dir = _validate_output_dir(output_dir)
        if not prompt or not prompt.strip():
            raise ValueError("[NV_SeedanceNativeChunkedLoop] prompt is empty.")

        cfg = _validate_upload_config_v2(upload_config)
        content_items = cfg["content"]
        image_content_items = [
            c for c in content_items if isinstance(c, dict) and c.get("kind") == "image"
        ]
        n_images = len(image_content_items)
        # validator already enforces ≥1 image; this is a defensive check.
        if n_images == 0:
            raise RuntimeError(
                "[NV_SeedanceNativeChunkedLoop] internal: validator passed but "
                "no image refs found in config content. Should not happen."
            )

        # Resolve ark- API key (matches NV_SeedanceNativeRefVideo_V2 pattern).
        # resolve_api_key raises RuntimeError if no key found. In debug_mode
        # we skip resolution entirely since no API calls will be made.
        if debug_mode:
            resolved_key = ""
        else:
            resolved_key = resolve_api_key(api_key, provider="volcengine")

        total_input_frames = int(images.shape[0])
        if total_input_frames < 1:
            raise ValueError("[NV_SeedanceNativeChunkedLoop] images is empty.")

        # Pixel-budget guard once — H,W constant across chunks
        images, did_resize, (img_h, img_w) = _clamp_frames_to_pixel_budget(images)
        if did_resize:
            print(
                f"[NV_SeedanceNativeChunkedLoop] Input clamped to {img_w}x{img_h} "
                f"to fit Volcengine pixel budget."
            )

        # Plan chunks (4-15s integer durations)
        chunk_ranges = _plan_seedance_chunks(total_input_frames, fps, target_chunk_seconds)
        chunk_count = len(chunk_ranges)
        prefix = _preflight_collision_check(out_dir, prefix, chunk_count, on_filename_collision)

        if slowdown_factor == 1:
            restore_target: int | None = None
        else:
            restore_target = _compute_original_from_slowed(total_input_frames, slowdown_factor)

        chunk_durations = [
            _chunk_duration_seconds(s, e, fps) for (s, e) in chunk_ranges
        ]

        model_id = _MODELS[model]
        # Auto-inject @Image1..N / @Video1 tags (or Chinese bracket form if
        # prompt contains CJK) — mirrors NV_SeedanceNativeRefVideo_V2 single-
        # shot parity. has_video=True because every chunk uploads a per-chunk
        # ref video. n_images = shared ref images from the config.
        final_prompt = _auto_inject_tags(prompt.strip(), n_images=n_images, has_video=True)
        tag_injected = final_prompt != prompt.strip()
        if tag_injected:
            print(f"[NV_SeedanceNativeChunkedLoop] @-tags auto-injected into final_prompt")

        # Pre-loop summary
        mode_label = "DEBUG (DRY RUN)" if debug_mode else "LIVE"
        save_pattern = f"{prefix}.mp4" if chunk_count == 1 else f"{prefix}_chunkNN.mp4"
        print(
            f"[NV_SeedanceNativeChunkedLoop {mode_label}] Planned {chunk_count} "
            f"chunk(s) of input ({total_input_frames} frames @ {fps}fps, "
            f"{total_input_frames/fps:.2f}s total). Model={model_id}, "
            f"res={resolution}, ratio={ratio}. Refs: {n_images} image(s) "
            f"(shared from Mode C config), +1 ref video per chunk. "
            f"Output → {out_dir}/{save_pattern}"
        )
        for i, ((s, e), d) in enumerate(zip(chunk_ranges, chunk_durations), 1):
            print(
                f"  chunk {i:02d}: frames [{s}:{e}] = {e-s} frames "
                f"({(e-s)/fps:.2f}s slice → requesting {d}s output)"
            )
        if debug_mode:
            dump_note = (
                " dump_request JSON files DO still write when dump_request=True."
                if dump_request else ""
            )
            print(
                "[NV_SeedanceNativeChunkedLoop DEBUG] No API calls, no uploads, "
                "no per-chunk MP4 saves, no charges. Chunk frames passed "
                "through as mock Seedance output for chunking + retime math "
                f"verification.{dump_note}"
            )
        if slowdown_factor > 1:
            print(
                f"[NV_SeedanceNativeChunkedLoop] slowdown_factor={slowdown_factor} → "
                f"will restore joined output to {restore_target} frames after concat."
            )
        if max_concurrent > 1:
            print(
                f"[NV_SeedanceNativeChunkedLoop] PARALLEL dispatch will fire "
                f"up to {max_concurrent} chunks concurrently. Per-chunk poll "
                f"timeout {poll_timeout_s:.0f}s. In-flight chunks are "
                f"non-cancellable once submitted (Volcengine has no remote "
                f"cancel — server-side task continues regardless of polling)."
            )

        # ---------- Phase 1: per-chunk processing (sequential or parallel) ----------
        trimmed_outputs: list[torch.Tensor] = []
        per_chunk_metadata: list[dict] = []
        per_chunk_task_ids: list[str] = []
        total_estimated_usd = 0.0
        failure_info: dict | None = None
        chunk_output_fps_observed: list[float] = []

        # In-flight task IDs: chunk_idx → task_id. Populated by the
        # on_task_submitted callback BEFORE polling each chunk, removed
        # after the chunk completes successfully. On interrupt / failure
        # mid-flight, this dict captures every Volcengine task that has
        # been submitted but not yet recovered locally — written to a
        # recovery sidecar before re-raising. Multi-task safe under
        # asyncio (single-threaded; mutations happen at await boundaries
        # in the dispatch closures, not concurrently).
        in_flight_task_ids: dict[int, str] = {}

        def _record_task_submitted(chunk_idx: int, task_id: str) -> None:
            in_flight_task_ids[chunk_idx] = task_id

        async def _process_chunk(
            chunk_idx_zero: int,
            start_frame: int,
            end_frame: int,
            duration_s: int,
        ) -> tuple[dict, torch.Tensor, float, str]:
            chunk_idx = chunk_idx_zero + 1
            chunk_label = f"chunk {chunk_idx:02d}/{chunk_count:02d}"
            _check_interrupt()

            entry: dict = {
                "chunk_index": chunk_idx,
                "filename": _chunk_filename(prefix, chunk_idx, chunk_count=chunk_count),
                "input_frame_range": [start_frame, end_frame],
                "input_frame_count": end_frame - start_frame,
                "requested_duration_s": duration_s,
                "status": "in_progress",
            }

            chunk_raw = images[start_frame:end_frame]

            if debug_mode:
                if dump_request:
                    dump_path = out_dir / f"{prefix}_chunk{chunk_idx:02d}_request.json"
                    dump_payload = {
                        "_meta": {
                            "endpoint": f"{_API_BASE}{_CREATE_PATH}",
                            "method": "POST",
                            "chunk_label": chunk_label,
                            "debug_mode": True,
                            "note": (
                                "video_url + image_url are placeholders (no "
                                "actual upload in debug mode). Counts + "
                                "structure reflect live behavior."
                            ),
                        },
                        "model": model_id,
                        "resolution": resolution,
                        "ratio": ratio,
                        "duration": duration_s,
                        "generate_audio": generate_audio,
                        "watermark": watermark,
                        "return_last_frame": return_last_frame,
                        "seed": seed,
                        "prompt": final_prompt,
                        "ref_image_count": n_images,
                        "ref_image_urls": [f"DEBUG_NO_UPLOAD_ref_{i+1}" for i in range(n_images)],
                        "chunk_video_url": "DEBUG_NO_UPLOAD",
                    }
                    try:
                        dump_path.parent.mkdir(parents=True, exist_ok=True)
                        dump_path.write_text(
                            json.dumps(dump_payload, indent=2, default=str), encoding="utf-8"
                        )
                        print(f"[NV_SeedanceNativeChunkedLoop {chunk_label}] dump_request (debug) → {dump_path}")
                    except OSError as _e:
                        print(f"[NV_SeedanceNativeChunkedLoop {chunk_label}] dump_request write failed: {_e}")
                output_images = chunk_raw
                output_fps_val = float(fps)
                task_id = "DEBUG_NO_API"
                chunk_api_meta = {
                    "task_id": task_id,
                    "result_video_url": None,
                    "duration_s": duration_s,
                    "input_frames": int(chunk_raw.shape[0]),
                    "input_fps": fps,
                    "input_resolution": f"{img_w}x{img_h}",
                    "output_frames": int(chunk_raw.shape[0]),
                    "output_fps": output_fps_val,
                    "output_resolution": f"{img_w}x{img_h}",
                    "status": "DEBUG_OK",
                    "token_usage": {
                        "completion_tokens": None,
                        "total_tokens": None,
                        "cost_estimate_usd": None,
                        "cost_formula": "DEBUG_NO_TOKEN_USAGE",
                    },
                    "timing": {"upload_sec": 0, "api_processing_sec": 0, "download_sec": 0, "total_sec": 0},
                    "debug_mode": True,
                }
                cost_usd = 0.0
            else:
                # LIVE: real API call to Volcengine. on_task_submitted fires
                # after POST returns with task_id, BEFORE polling — so
                # interrupt-during-poll has the task_id available for
                # recovery via the in-flight sidecar.
                output_images, chunk_api_meta, output_fps_val, token_usage, task_id = (
                    await _seedance_native_chunk_api_call(
                        cls,
                        chunk_raw,
                        encode_fps=fps,
                        final_prompt=final_prompt,
                        image_content_items=image_content_items,
                        model_id=model_id,
                        resolution=resolution,
                        ratio=ratio,
                        duration_s=duration_s,
                        generate_audio=generate_audio,
                        watermark=watermark,
                        seed=seed,
                        return_last_frame=return_last_frame,
                        api_key=resolved_key,
                        poll_interval_s=poll_interval_s,
                        poll_timeout_s=poll_timeout_s,
                        chunk_label=chunk_label,
                        dump_request_to=(
                            out_dir / f"{prefix}_chunk{chunk_idx:02d}_request.json"
                            if dump_request else None
                        ),
                        on_task_submitted=lambda tid, _idx=chunk_idx: _record_task_submitted(_idx, tid),
                    )
                )
                cost_usd = token_usage.get("cost_estimate_usd") or 0.0
                # NOTE: Do NOT pop from in_flight_task_ids here. The
                # remote Volcengine task has succeeded, but local save
                # (below) could still fail (disk full, permission, etc.),
                # in which case we want this chunk's task_id to appear
                # in the recovery sidecar so the user can re-download
                # via NV_SeedanceFetchTask. Pop only AFTER local save
                # completes (or in debug, before save block since debug
                # never enters in_flight in the first place).

            entry["api"] = chunk_api_meta
            entry["task_id"] = task_id  # Surface explicitly for easy recovery

            if debug_mode:
                entry["saved_path"] = None
                entry["saved_bytes"] = 0
                entry["debug_mode"] = True
                entry["status"] = "ok"
                print(
                    f"[NV_SeedanceNativeChunkedLoop {chunk_label} DEBUG] mock OK — "
                    f"{int(output_images.shape[0])} mock frames passed through, "
                    f"NO file written"
                )
            else:
                target = out_dir / _chunk_filename(prefix, chunk_idx, chunk_count=chunk_count)
                bytes_written = await _download_url_to_file(
                    chunk_api_meta["result_video_url"], target
                )
                entry["saved_path"] = str(target)
                entry["saved_bytes"] = bytes_written
                entry["status"] = "ok"
                # Local save succeeded — now safe to remove from in-flight
                # set. If save had failed (OSError etc.), exception would
                # have propagated to outer dispatch with this chunk's
                # task_id still in in_flight_task_ids for the recovery
                # sidecar.
                in_flight_task_ids.pop(chunk_idx, None)
                print(
                    f"[NV_SeedanceNativeChunkedLoop {chunk_label}] OK — "
                    f"{int(output_images.shape[0])} frames @ {output_fps_val:.2f}fps, "
                    f"saved {bytes_written/1024/1024:.2f} MB to {target.name} "
                    f"(task_id={task_id}, cost≈${cost_usd:.4f})"
                )

            return entry, output_images, cost_usd, task_id

        # Dispatch — sequential vs parallel.
        # InterruptProcessingException is re-raised up to ComfyUI's cancel
        # handler in both paths (R1-R4 hardening from D-385 follow-up).
        if max_concurrent == 1:
            for chunk_idx_zero, ((start_frame, end_frame), duration_s) in enumerate(
                zip(chunk_ranges, chunk_durations)
            ):
                chunk_idx = chunk_idx_zero + 1
                chunk_label = f"chunk {chunk_idx:02d}/{chunk_count:02d}"
                try:
                    entry, output_images, cost_usd, task_id = await _process_chunk(
                        chunk_idx_zero, start_frame, end_frame, duration_s
                    )
                    per_chunk_metadata.append(entry)
                    trimmed_outputs.append(output_images)
                    per_chunk_task_ids.append(task_id)
                    total_estimated_usd += cost_usd
                    chunk_output_fps_observed.append(entry["api"]["output_fps"])
                except _InterruptProcessingException:
                    print(
                        f"[NV_SeedanceNativeChunkedLoop {chunk_label}] "
                        f"INTERRUPTED by ComfyUI Cancel — re-raising up to "
                        f"executor ({len(per_chunk_metadata)} chunks recorded "
                        f"so far). Any in-flight Volcengine task continues "
                        f"server-side; recover via NV_SeedanceFetchTask."
                    )
                    _write_in_flight_recovery_sidecar(
                        out_dir, prefix, in_flight_task_ids,
                        reason="sequential dispatch interrupted by ComfyUI Cancel",
                    )
                    raise
                except Exception as e:
                    fail_entry = {
                        "chunk_index": chunk_idx,
                        "filename": _chunk_filename(prefix, chunk_idx, chunk_count=chunk_count),
                        "input_frame_range": [start_frame, end_frame],
                        "input_frame_count": end_frame - start_frame,
                        "requested_duration_s": duration_s,
                        "status": "failed",
                        "error_type": e.__class__.__name__,
                        "error_message": str(e),
                    }
                    per_chunk_metadata.append(fail_entry)
                    failure_info = {
                        "failed_chunk_index": chunk_idx,
                        "failed_chunk_label": chunk_label,
                        "error_type": e.__class__.__name__,
                        "error_message": str(e),
                    }
                    print(
                        f"[NV_SeedanceNativeChunkedLoop {chunk_label}] FAILED — "
                        f"{e.__class__.__name__}: {e}"
                    )
                    break
        else:
            chunk_factories = [
                (lambda i=i, s=s, e=e, d=d: _process_chunk(i, s, e, d))
                for i, ((s, e), d) in enumerate(zip(chunk_ranges, chunk_durations))
            ]
            results = await _run_chunks_concurrent(chunk_factories, max_concurrent)

            # FIRST PASS: scan for InterruptProcessingException — see D-386
            # for full rationale (gather captures it as a regular result; we
            # must bubble it up to ComfyUI's Cancel handler).
            for chunk_idx_zero, result in enumerate(results):
                if isinstance(result, _InterruptProcessingException):
                    print(
                        f"[NV_SeedanceNativeChunkedLoop] INTERRUPTED by "
                        f"ComfyUI Cancel (chunk "
                        f"{chunk_idx_zero + 1:02d}/{chunk_count:02d} raised "
                        f"first) — re-raising. In-flight Volcengine tasks "
                        f"continue server-side; recover via "
                        f"NV_SeedanceFetchTask once interrupt settles."
                    )
                    _write_in_flight_recovery_sidecar(
                        out_dir, prefix, in_flight_task_ids,
                        reason="parallel dispatch interrupted by ComfyUI Cancel",
                    )
                    raise result

            for chunk_idx_zero, result in enumerate(results):
                chunk_idx = chunk_idx_zero + 1
                start_frame, end_frame = chunk_ranges[chunk_idx_zero]
                duration_s = chunk_durations[chunk_idx_zero]
                chunk_label = f"chunk {chunk_idx:02d}/{chunk_count:02d}"
                if isinstance(result, BaseException):
                    fail_entry = {
                        "chunk_index": chunk_idx,
                        "filename": _chunk_filename(prefix, chunk_idx, chunk_count=chunk_count),
                        "input_frame_range": [start_frame, end_frame],
                        "input_frame_count": end_frame - start_frame,
                        "requested_duration_s": duration_s,
                        "status": "failed",
                        "error_type": result.__class__.__name__,
                        "error_message": str(result),
                    }
                    per_chunk_metadata.append(fail_entry)
                    if failure_info is None:
                        failure_info = {
                            "failed_chunk_index": chunk_idx,
                            "failed_chunk_label": chunk_label,
                            "error_type": result.__class__.__name__,
                            "error_message": str(result),
                        }
                    print(
                        f"[NV_SeedanceNativeChunkedLoop {chunk_label}] FAILED — "
                        f"{result.__class__.__name__}: {result}"
                    )
                else:
                    entry, output_images, cost_usd, task_id = result
                    per_chunk_metadata.append(entry)
                    trimmed_outputs.append(output_images)
                    per_chunk_task_ids.append(task_id)
                    total_estimated_usd += cost_usd
                    chunk_output_fps_observed.append(entry["api"]["output_fps"])

        # ---------- Phase 2: aggregate & restore (or raise) ----------
        elapsed = round(time.time() - t_overall_start, 1)
        succeeded = [m for m in per_chunk_metadata if m.get("status") == "ok"]
        failed = [m for m in per_chunk_metadata if m.get("status") == "failed"]

        fps_warning = None
        if len(set(chunk_output_fps_observed)) > 1:
            fps_warning = (
                f"per-chunk output_fps varies across chunks: "
                f"{chunk_output_fps_observed}. Concat is still valid (frame "
                f"count is additive) but timeline pacing of the joined output "
                f"will be uneven. The retime restore is fps-agnostic "
                f"(proportional select on frame count), so the final frame "
                f"count is still correct."
            )
            print(f"[NV_SeedanceNativeChunkedLoop] WARN: {fps_warning}")

        task_ids_json = json.dumps(per_chunk_task_ids)

        meta_json = json.dumps({
            "summary": {
                "debug_mode": debug_mode,
                "max_concurrent": max_concurrent,
                "chunk_count_planned": chunk_count,
                "chunk_count_succeeded": len(succeeded),
                "chunk_count_failed": len(failed),
                "total_estimated_usd": round(total_estimated_usd, 4),
                "wall_seconds": elapsed,
                "output_dir": str(out_dir),
                "filename_prefix": prefix,
                "input_frames": total_input_frames,
                "input_fps": fps,
                "input_resolution": f"{img_w}x{img_h}",
                "input_was_resized": did_resize,
                "n_reference_images": n_images,
                "mode": cfg.get("mode"),
                "model": model_id,
                "resolution": resolution,
                "ratio": ratio,
                "target_chunk_seconds": target_chunk_seconds,
                "slowdown_factor": slowdown_factor,
                "restore_target_frames": restore_target,
                "restore_applied": False,
                "chunk_output_fps_observed": chunk_output_fps_observed,
                "fps_heterogeneity_warning": fps_warning,
                "task_ids": per_chunk_task_ids,
            },
            "chunks": per_chunk_metadata,
            "failure": failure_info,
        }, indent=2)

        if failure_info:
            saved_paths = [m["saved_path"] for m in succeeded]
            fail_count = len(failed)
            multi_fail_note = (
                f" ({fail_count} chunks failed total — see per_chunk_metadata "
                f"for the full failure list)"
                if fail_count > 1
                else ""
            )
            # If any sibling chunks were still in-flight when the failure
            # was aggregated, write the recovery sidecar so the user can
            # recover those tasks via NV_SeedanceFetchTask.
            _write_in_flight_recovery_sidecar(
                out_dir, prefix, in_flight_task_ids,
                reason="chunk(s) failed with sibling task(s) still in-flight",
            )
            raise RuntimeError(
                f"[NV_SeedanceNativeChunkedLoop] Failed at "
                f"{failure_info['failed_chunk_label']}: "
                f"{failure_info['error_type']}: {failure_info['error_message']}"
                f"{multi_fail_note}. "
                f"{len(succeeded)}/{chunk_count} chunks saved to disk: "
                f"{saved_paths}. completed task_ids: {per_chunk_task_ids}. "
                f"In-flight task_ids (recoverable via NV_SeedanceFetchTask): "
                f"{dict(in_flight_task_ids) if in_flight_task_ids else 'none'}. "
                f"Inspect per_chunk_metadata for full breakdown."
            )

        # All chunks succeeded — concat outputs in chunk order
        joined = torch.cat(trimmed_outputs, dim=0) if len(trimmed_outputs) > 1 else trimmed_outputs[0]
        joined_count = int(joined.shape[0])
        print(
            f"[NV_SeedanceNativeChunkedLoop] Joined {len(trimmed_outputs)} chunks → "
            f"{joined_count} concatenated frames"
        )

        if restore_target is not None:
            restored = _restore_proportional(joined, restore_target)
            print(
                f"[NV_SeedanceNativeChunkedLoop] Restored {joined_count} → "
                f"{restored.shape[0]} frames (slowdown_factor={slowdown_factor})"
            )
            joined = restored
            patched = json.loads(meta_json)
            patched["summary"]["restore_applied"] = True
            patched["summary"]["restored_frames"] = int(restored.shape[0])
            meta_json = json.dumps(patched, indent=2)

        done_label = "DONE (DEBUG, no charges)" if debug_mode else "DONE"
        usd_label = "would-bill " if debug_mode else ""
        print(
            f"[NV_SeedanceNativeChunkedLoop] {done_label} in {elapsed}s — "
            f"{chunk_count} chunks, {usd_label}${round(total_estimated_usd, 4)} total"
        )

        if succeeded and succeeded[0].get("api"):
            output_fps_val = float(succeeded[0]["api"].get("output_fps", fps))
        else:
            output_fps_val = float(fps)
        output_frames_val = int(joined.shape[0])

        return IO.NodeOutput(
            joined,
            output_fps_val,
            output_frames_val,
            final_prompt,
            task_ids_json,
            meta_json,
            round(total_estimated_usd, 4),
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_SeedanceNativeChunkedLoop_V2": NV_SeedanceNativeChunkedLoop_V2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SeedanceNativeChunkedLoop_V2": "NV Seedance Native Chunked Loop V2",
}
