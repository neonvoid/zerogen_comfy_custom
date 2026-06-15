"""NV BytePlus Seedance Gen — native generation on the BytePlus INTERNATIONAL
endpoint (ap-southeast), consuming asset:// refs from the BytePlus register nodes.

This is the generation half of the console-free native pipeline. It pairs with
NV_ByteplusImageAssetRegister / NV_ByteplusImageBatchRegister: their asset://
output wires straight into `ref_image_asset_urls` here, and this node submits the
generation against the SAME library/endpoint the assets were registered on.

Platform note (the whole reason this is a separate node from NV_SeedanceNativeRefVideo_V2):
- NV_SeedanceNativeRefVideo_V2 → ark.cn-beijing.volces.com, `doubao-` models = Volcengine MAINLAND.
- THIS node → ark.ap-southeast.bytepluses.com, `dreamina-` models = BytePlus INTERNATIONAL.
These are different platforms with SEPARATE asset libraries; a BytePlus asset://
id is only valid on the international endpoint. Runtime-validated end-to-end
2026-06-11 (probe D:/tmp/byteplus_gen_smoke.py: asset:// ref → 720p succeeded).

Refs are URL strings (asset:// / HTTPS / data: URI) — NOT a tensor upload_config.
That matches how the register nodes and the Moyu chunked loop pass refs, so this
slots into the same graph position as the Moyu workflow's generation node.

Self-contained (own copies of the post/poll/error infra) per the operator's
"BytePlus stays a separate node group" constraint — the mainland nodes are
untouched.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from io import BytesIO

import aiohttp
import numpy as np
import torch
from PIL import Image

from comfy_api.latest import IO

from .api_keys import resolve_api_key

# Note: comfy_api_nodes.util download helpers are imported lazily inside execute()
# — they pull in the `utils` package chain that only resolves in a live ComfyUI
# process, so deferring keeps this module importable for schema/unit smoke.

try:
    from comfy.model_management import throw_exception_if_processing_interrupted as _check_interrupt
except ImportError:
    def _check_interrupt() -> None:
        pass


# ---------------------------------------------------------------------------
# Constants — BytePlus INTERNATIONAL (ap-southeast)
# ---------------------------------------------------------------------------

_API_BASE = "https://ark.ap-southeast.bytepluses.com/api/v3"
_CREATE_PATH = "/contents/generations/tasks"
_STATUS_PATH = "/contents/generations/tasks"

# Pro id validated live 2026-06-11. Fast id is INFERRED from the mainland naming
# pattern (doubao-...-fast-... → dreamina-...-fast-...) and NOT yet validated —
# the tooltip flags this; first Fast run confirms or 404s (one-line fix).
_MODELS = {
    "Seedance 2.0 Pro": "dreamina-seedance-2-0-260128",
    "Seedance 2.0 Fast": "dreamina-seedance-2-0-fast-260128",
}

_RESOLUTIONS = ["480p", "720p", "1080p"]
_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]

_MAX_REF_IMAGES = 9   # Seedance Mode C reference-image cap (matches the batch register)


class _TransientPollHTTPError(RuntimeError):
    """A 5xx during status polling — retryable; the server-side job continues."""


# ---------------------------------------------------------------------------
# Helpers (self-contained copies; differ from the mainland node only in the
# base URL + a couple of international-specific error hints)
# ---------------------------------------------------------------------------


def _auto_inject_tags(prompt: str, n_images: int, has_video: bool) -> str:
    """Prefix @Image1..N / @Video1 if the prompt has no ref tags (EN or CN form).

    Injects even when the prompt is EMPTY (img2vid: refs but no text) — otherwise
    the refs reach the model with no @ImageN mapping at all (R0 review HIGH).
    English tag detection is case-insensitive so a user's `@image1` isn't
    treated as "no tag" and duplicated (R0 review).
    """
    p = prompt or ""
    has_image_tag = bool(re.search(r"@Image\s?\d+", p, re.IGNORECASE)) or bool(re.search(r"\[图\s?\d+\]", p))
    has_video_tag = bool(re.search(r"@Video\s?\d+", p, re.IGNORECASE)) or bool(re.search(r"\[视频\s?\d+\]", p))
    has_cjk = bool(re.search(r"[一-鿿]", p))
    parts: list[str] = []
    if has_video and not has_video_tag:
        parts.append("[视频1]" if has_cjk else "@Video1")
    if n_images > 0 and not has_image_tag:
        parts.extend((f"[图{i}]" if has_cjk else f"@Image{i}") for i in range(1, n_images + 1))
    if not parts:
        return p
    return f"{' '.join(parts)} {p}".strip()


def _classify_http_error(status_code: int, body_text: str) -> str:
    hints = []
    body_low = body_text.lower()
    if status_code == 401:
        hints.append("401 Unauthorized — check the Bearer ARK_API_KEY value (env ARK_API_KEY / VOLCENGINE_ARK_API_KEY / .env)")
    elif status_code == 403:
        hints.append("403 Forbidden — check the BytePlus account balance / Seedance 2.0 access on the ap-southeast project")
    elif status_code == 429:
        hints.append("429 rate limited — reduce request frequency or raise quota")
    elif status_code == 400:
        if "sensitive" in body_low or "privacyinformation" in body_low or "real person" in body_low:
            hints.append(
                "400 — INPUT content gate. Real-face refs are input-gated on the native endpoint; "
                "register the face via NV_ByteplusImageAssetRegister (Virtual Portrait / AIGC group) "
                "and pass the asset:// id instead of a direct image URL."
            )
        elif "asset" in body_low and ("not found" in body_low or "notfound" in body_low):
            hints.append(
                "400 — asset:// not found on THIS endpoint. The asset must be Active in the SAME project "
                "and registered on the international (ap-southeast) library, not Moyu/mainland."
            )
        elif "size" in body_low or "too large" in body_low:
            hints.append("400 — payload too large (prefer asset:// / URL refs over base64)")
        else:
            hints.append("400 — request rejected; see raw message below")
    elif 500 <= status_code < 600:
        hints.append(f"{status_code} transient upstream — retry may help")
    return " | ".join(hints) if hints else f"HTTP {status_code}"


def _parse_ref_urls(multiline: str, singular: str) -> list[str]:
    """Resolve image refs: multi-line list wins over the singular widget.

    Returns asset:// / HTTPS / data: URI strings in order. Raises if BOTH a
    non-empty multiline list AND a non-empty singular are given (ambiguous —
    mirrors the Moyu chunked loop's mutual-exclusion guard).
    """
    multi = [ln.strip() for ln in (multiline or "").splitlines() if ln.strip()]
    sing = (singular or "").strip()
    if multi and sing:
        raise ValueError(
            "Provide refs via EITHER ref_image_asset_urls (multi-line) OR "
            "ref_image_asset_url (singular), not both."
        )
    if multi:
        return multi
    if sing:
        return [sing]
    return []


def _build_content(final_prompt: str, image_urls: list[str], video_url: str) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": final_prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"})
    v = (video_url or "").strip()
    if v:
        content.append({"type": "video_url", "video_url": {"url": v}, "role": "reference_video"})
    return content


async def _post_task(session, api_key, payload):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    async with session.post(f"{_API_BASE}{_CREATE_PATH}", json=payload, headers=headers) as resp:
        body_text = await resp.text()
        if resp.status != 200:
            hint = _classify_http_error(resp.status, body_text)
            raise RuntimeError(f"BytePlus Seedance task creation failed: {hint}\nRaw response:\n{body_text}")
        try:
            return json.loads(body_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"BytePlus Seedance task creation non-JSON: {e}\nBody: {body_text[:500]}")


async def _get_task(session, api_key, task_id):
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{_API_BASE}{_STATUS_PATH}/{task_id}"
    async with session.get(url, headers=headers) as resp:
        body_text = await resp.text()
        if resp.status != 200:
            hint = _classify_http_error(resp.status, body_text)
            # 5xx during poll is transient — let the retry loop handle it (R0 review)
            # rather than forfeit an otherwise-running paid job on a single blip.
            if 500 <= resp.status < 600:
                raise _TransientPollHTTPError(f"BytePlus Seedance status {resp.status}: {hint}")
            raise RuntimeError(f"BytePlus Seedance status lookup failed: {hint}\nRaw: {body_text}")
        try:
            return json.loads(body_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"BytePlus Seedance status non-JSON: {e}\nBody: {body_text[:500]}")


# Transient network failures during long polls must NOT forfeit a server-side job.
_TRANSIENT_POLL_ERRORS = (
    aiohttp.ClientConnectionError,
    aiohttp.ClientPayloadError,
    asyncio.TimeoutError,
    ConnectionResetError,
    ConnectionError,
    _TransientPollHTTPError,   # 5xx status responses (R0 review)
)
_MAX_CONSECUTIVE_POLL_FAILURES = 10
_POLL_RETRY_BACKOFF_CAP_S = 30.0


async def _poll_task(session, api_key, task_id, interval, timeout):
    deadline = time.time() + timeout
    last_status = None
    consecutive_failures = 0
    while True:
        _check_interrupt()
        try:
            resp = await _get_task(session, api_key, task_id)
        except _TRANSIENT_POLL_ERRORS as e:
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_POLL_FAILURES:
                raise RuntimeError(
                    f"BytePlus Seedance poll failed {consecutive_failures}× consecutively — likely a real "
                    f"outage. task_id={task_id}. Last error: {type(e).__name__}: {e}. "
                    f"Re-run with NV_SeedanceFetchTask once connectivity recovers."
                ) from e
            backoff = min(_POLL_RETRY_BACKOFF_CAP_S, 2.0 * consecutive_failures)
            print(f"[NV_ByteplusSeedanceGen] poll #{consecutive_failures} transient error "
                  f"({type(e).__name__}: {e}); retrying in {backoff:.1f}s. task_id={task_id}")
            elapsed = 0.0
            while elapsed < backoff:
                _check_interrupt()
                step = min(0.5, backoff - elapsed)
                await asyncio.sleep(step)
                elapsed += step
            deadline += backoff  # retry sleeps don't count against job-completion budget
            continue
        consecutive_failures = 0
        status = resp.get("status")
        if status != last_status:
            print(f"[NV_ByteplusSeedanceGen] status: {status}")
            last_status = status
        if status in ("succeeded", "failed", "expired", "cancelled"):
            return resp
        if time.time() > deadline:
            raise RuntimeError(
                f"BytePlus Seedance task poll timed out after {timeout:.0f}s (last: {status}). task_id={task_id}"
            )
        elapsed = 0.0
        while elapsed < interval:
            _check_interrupt()
            step = min(0.5, interval - elapsed)
            await asyncio.sleep(step)
            elapsed += step


def _token_usage_summary(resp) -> dict:
    """Report token usage. International billing is token-based; we don't ship a
    price table yet (would need the dreamina token rate), so cost is left null
    with a pointer rather than reusing the mainland CNY/sec table (wrong model)."""
    usage = resp.get("usage") or {}
    return {
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_estimate_usd": None,
        "cost_note": "BytePlus international is token-billed; see the ModelArk console for the dreamina rate.",
    }


# ---------------------------------------------------------------------------
# Duration (smart timing) + retime (slowdown restore) — pure helpers
# ---------------------------------------------------------------------------

_DURATION_MODES = ["manual", "model_auto", "auto_from_ref"]


def _resolve_duration(duration_mode: str, duration: int, ref_duration_s: float) -> tuple[int, str]:
    """Resolve the API `duration`. Returns (api_duration, source_note). RAISES on
    an unknown mode or an out-of-range resolved duration (R0 review: validate
    centrally for EVERY mode, not just manual — the auto_from_ref fallback could
    otherwise pass a 0-3 slider value straight to the API).

    - manual        → the slider value (must be -1 or 4-15).
    - model_auto    → -1 (let the model pick).
    - auto_from_ref → ceil(ref_duration_s) clamped 4-15, so the output length
      matches the reference clip (avoids Seedance inventing extra time past the
      ref motion — the "weird cuts" failure). Falls back to the (validated)
      slider value when no ref_duration_s is supplied.
    """
    if duration_mode not in _DURATION_MODES:
        raise ValueError(f"unknown duration_mode {duration_mode!r}; expected one of {_DURATION_MODES}.")
    if duration_mode == "model_auto":
        return -1, "model_auto (-1)"
    if duration_mode == "auto_from_ref" and ref_duration_s and ref_duration_s > 0:
        d = max(4, min(15, math.ceil(float(ref_duration_s))))
        return d, f"auto_from_ref ({float(ref_duration_s):.2f}s → ceil → {d}s, clamped 4-15)"
    # manual, or auto_from_ref with no ref duration → use the slider, but validate.
    if duration != -1 and not (4 <= duration <= 15):
        raise ValueError(f"duration must be -1 (model-auto) or 4-15 seconds; got {duration}.")
    note = f"manual ({duration}s)" if duration_mode == "manual" else f"auto_from_ref → manual fallback ({duration}s): no ref_duration_s"
    return duration, note


def _measure_video_duration(video) -> float:  # noqa: ANN001 — Input.Video duck-typed
    """Best-effort source duration (seconds) from a VIDEO object. 0.0 if absent
    or unmeasurable. Lets a node measure the LOCAL source clip's length itself —
    independent of how the ref reaches the API (works for pre-registered
    asset:// URLs + cache hits, where there's no per-run measurement otherwise)."""
    if video is None:
        return 0.0
    try:
        return float(video.get_duration())
    except Exception:
        try:
            c = video.get_components()
            n = int(c.images.shape[0])
            fps = float(c.frame_rate)
            return n / fps if fps else 0.0
        except Exception:
            return 0.0


def _url_path_tail(url: str | None, n: int = 60) -> str | None:
    """Tail of the URL PATH only (drop the query string — presigned signature
    material lives in the query; R0 review). For debug 'which file' visibility."""
    if not url:
        return None
    return url.split("?", 1)[0][-n:]


def _restore_proportional(images, original_count: int):
    """Pick `original_count` evenly-spaced frames (copy of the chunked-loop op,
    kept local so this module stays self-contained)."""
    available = int(images.shape[0])
    if original_count < 1:
        original_count = 1
    if available <= 1 or original_count >= available:
        return images
    if original_count == 1:
        return images[0:1]
    step = (available - 1) / (original_count - 1)
    indices = [min(round(i * step), available - 1) for i in range(original_count)]
    return images[indices]


def _retime_restore(frames, slowdown_factor: int):
    """Speed output frames back up by `slowdown_factor` (the factor the source
    was slowed by before generation). Returns (frames, retimed_bool).

    Lenient — works for any Seedance output frame count by evenly resampling to
    round(N / factor) frames. factor<=1 is a pass-through.
    """
    if not slowdown_factor or slowdown_factor <= 1:
        return frames, False
    n = int(frames.shape[0])
    target = max(1, round(n / slowdown_factor))
    if target >= n:
        return frames, False
    return _restore_proportional(frames, target), True


# ---------------------------------------------------------------------------
# Shared generation core — used by the single node AND the multi-job
# ---------------------------------------------------------------------------


async def _generate_one(
    *,
    cls,
    final_prompt: str,
    image_urls: list,
    video_url: str,
    model_id: str,
    resolution: str,
    ratio: str,
    api_duration: int,
    generate_audio: bool,
    watermark: bool,
    seed: int,
    return_last_frame: bool,
    slowdown_factor: int,
    poll_interval_s: float,
    poll_timeout_s: float,
    resolved_key: str,
    log_tag: str = "NV_ByteplusSeedanceGen",
) -> dict:
    """Submit one gen task → poll → download → decode → retime-restore.

    Returns a result dict with the artifacts (frames/fps/last_frame/video/
    task_id/final_resp + retime info). Raises on non-succeeded status or a
    download failure (the caller decides hard-raise vs soft-fail). `cls` is the
    api-node context the download helpers need.
    """
    from comfy_api_nodes.util import download_url_to_video_output
    from comfy_api_nodes.util.download_helpers import download_url_to_bytesio

    content = _build_content(final_prompt, image_urls, video_url)
    payload: dict = {
        "model": model_id,
        "content": content,
        "resolution": resolution,
        "ratio": ratio,
        "duration": api_duration,
        "generate_audio": generate_audio,
        "watermark": watermark,
        "return_last_frame": return_last_frame,
    }
    if seed != -1:
        payload["seed"] = seed

    t_submit = time.time()
    session_timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=120)
    connector = aiohttp.TCPConnector(force_close=True, limit=8)
    session = aiohttp.ClientSession(timeout=session_timeout, connector=connector)
    try:
        create_resp = await _post_task(session, resolved_key, payload)
        task_id = create_resp.get("id")
        if not task_id:
            raise RuntimeError(f"Task creation returned no id. Raw: {create_resp}")
        print(f"[{log_tag}] Task submitted: {task_id}")
        final_resp = await _poll_task(session, resolved_key, task_id, poll_interval_s, poll_timeout_s)
    finally:
        await session.close()
        await asyncio.sleep(0.1)  # Windows ProactorEventLoop SSL cleanup tick
    t_done = time.time()

    status = final_resp.get("status")
    if status != "succeeded":
        err = final_resp.get("error") or {}
        code = err.get("code")
        msg = err.get("message")
        extra = ""
        if code and ("SensitiveContent" in str(code) or "PolicyViolation" in str(code)):
            extra = (
                " — OUTPUT content gate (post-generation, copyright/likeness). NOT the input gate and "
                "NOT asset-library-bypassable; it is face/IP-driven. See memory sd2_output_content_gate."
            )
        raise RuntimeError(
            f"BytePlus Seedance task ended status={status}. error.code={code!r} error.message={msg!r}.{extra} "
            f"task_id={task_id}"
        )

    resp_content = final_resp.get("content") or {}
    result_video_url = resp_content.get("video_url")
    last_frame_url = resp_content.get("last_frame_url")
    if not result_video_url:
        raise RuntimeError(f"Task succeeded but content.video_url missing. Raw: {final_resp}")

    try:
        output_video = await download_url_to_video_output(result_video_url)
    except Exception as e:
        raise RuntimeError(
            f"BytePlus task SUCCEEDED but video download failed — the generation is billed and the video "
            f"exists server-side. task_id={task_id}. Recover with NV_SeedanceFetchTask. "
            f"Error: {type(e).__name__}: {e}"
        ) from e

    try:
        components = output_video.get_components()
        out_images = components.images
        out_fps = float(components.frame_rate)
    except Exception as e:
        print(f"[{log_tag}] Warning: frame decode failed: {e}")
        out_images = torch.zeros(1, 64, 64, 3)
        out_fps = 0.0

    raw_frame_count = int(out_images.shape[0])
    out_images, retimed = _retime_restore(out_images, slowdown_factor)
    out_frames = int(out_images.shape[0])
    if retimed:
        print(f"[{log_tag}] retime restore: {raw_frame_count} → {out_frames} frames (slowdown_factor={slowdown_factor}).")

    last_frame_tensor = torch.zeros(1, 64, 64, 3)
    last_frame_fetched = False
    if return_last_frame and last_frame_url:
        try:
            png_bytes = BytesIO()
            await download_url_to_bytesio(last_frame_url, png_bytes, timeout=30, max_retries=3, cls=cls)
            png_bytes.seek(0)
            img = Image.open(png_bytes).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            last_frame_tensor = torch.from_numpy(arr).unsqueeze(0)
            last_frame_fetched = True
        except Exception as e:
            print(f"[{log_tag}] Warning: last_frame fetch failed: {e}")

    return {
        "frames": out_images,
        "fps": out_fps,
        "frame_count": out_frames,
        "raw_frame_count": raw_frame_count,
        "retimed": retimed,
        "last_frame": last_frame_tensor,
        "last_frame_fetched": last_frame_fetched,
        "video": output_video,
        "task_id": task_id,
        "final_resp": final_resp,
        "video_url": result_video_url,
        "last_frame_url": last_frame_url,
        "submit_to_done_s": round(t_done - t_submit, 1),
    }


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class NV_ByteplusSeedanceGen(IO.ComfyNode):
    """Generate a Seedance 2.0 video on the BytePlus international endpoint.

    Pair with the BytePlus register nodes: wire their asset:// output into
    `ref_image_asset_urls`. Single-shot (one generation task) — the chunked /
    multi-job variant is a separate node.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_ByteplusSeedanceGen",
            display_name="NV BytePlus Seedance Gen",
            category="NV_Utils/api",
            description=(
                "Generate a Seedance 2.0 video on the BytePlus INTERNATIONAL endpoint "
                "(ark.ap-southeast.bytepluses.com, dreamina- models). Consumes asset:// "
                "refs from NV_ByteplusImage(Batch)AssetRegister — the console-free "
                "real-face path. Refs are URL strings (asset:// / HTTPS / data:). Uses "
                "the Bearer ARK_API_KEY from env / .env."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Generation prompt. @Image1..N / @Video1 ref tags auto-injected if absent.",
                ),
                IO.String.Input(
                    "ref_image_asset_urls",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Newline-separated image refs (1-9), one per line, in @Image1..N order. "
                        "Wire NV_ByteplusImageBatchRegister.joined_urls here. Each is an asset:// id "
                        "(or HTTPS URL / data: URI). Real-face refs MUST be asset:// (direct image "
                        "URLs are input-gated). Takes priority over the singular widget."
                    ),
                    optional=True,
                ),
                IO.String.Input(
                    "ref_image_asset_url",
                    default="",
                    tooltip="Single image ref (asset:// / HTTPS / data:). For the one-ref case. Mutually exclusive with the multi-line list above.",
                    optional=True,
                ),
                IO.String.Input(
                    "ref_video_asset_url",
                    default="",
                    tooltip="Optional reference VIDEO (asset:// / HTTPS). Role = reference_video.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "model",
                    options=list(_MODELS.keys()),
                    default="Seedance 2.0 Pro",
                    tooltip="Pro id validated live. Fast id is inferred from mainland naming — validate before relying on it.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=_RESOLUTIONS,
                    default="720p",
                    tooltip="480p / 720p / 1080p. 720p is validated; 1080p support on dreamina is not yet runtime-confirmed — test it.",
                ),
                IO.Combo.Input(
                    "ratio",
                    options=_RATIOS,
                    default="adaptive",
                    tooltip="adaptive = model picks from refs.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=-1,
                    max=15,
                    step=1,
                    tooltip="Output seconds. 4-15, or -1 for model-auto.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Boolean.Input("generate_audio", default=True, tooltip="Produce synchronized audio."),
                IO.Boolean.Input("watermark", default=False, tooltip="Add watermark to output."),
                IO.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="-1 = random. Non-deterministic even at a fixed seed.",
                ),
                IO.Boolean.Input(
                    "return_last_frame",
                    default=True,
                    tooltip="Request the output's last frame as PNG (wires into a next chunk's first_frame).",
                ),
                IO.Float.Input("poll_interval_s", default=9.0, min=2.0, max=60.0, step=1.0, tooltip="Seconds between status polls."),
                IO.Float.Input(
                    "poll_timeout_s",
                    default=1500.0,
                    min=60.0,
                    max=7200.0,
                    step=30.0,
                    tooltip="Max seconds to wait. On timeout the task_id is logged — recover later with NV_SeedanceFetchTask.",
                ),
                IO.String.Input(
                    "api_key",
                    default="",
                    tooltip="Optional Bearer ARK_API_KEY override. Empty → env ARK_API_KEY / VOLCENGINE_ARK_API_KEY / .env.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "duration_mode",
                    options=_DURATION_MODES,
                    default="manual",
                    tooltip=(
                        "manual = use the duration slider. model_auto = send -1 (model picks). "
                        "auto_from_ref = output length = ceil(ref_duration_s) clamped 4-15, so the output "
                        "matches the reference clip (prevents Seedance inventing extra time past the ref "
                        "motion — the 'weird cuts' artifact). Wire ref_duration_s from NV_VIDEOINFO.duration."
                    ),
                    optional=True,
                ),
                IO.Video.Input(
                    "ref_video",
                    tooltip=(
                        "Optional LOCAL source video — the node measures its duration here for "
                        "duration_mode=auto_from_ref, so timing works even with pre-registered asset:// "
                        "refs (the asset:// URL is the API ref; this is just the local measurement "
                        "source). Wire the same clip you registered. Takes priority over ref_duration_s."
                    ),
                    optional=True,
                ),
                IO.Float.Input(
                    "ref_duration_s",
                    default=0.0,
                    min=0.0,
                    max=600.0,
                    step=0.01,
                    tooltip="Manual reference duration (seconds) fallback for auto_from_ref when no ref_video is wired (e.g. you only have the asset:// id + a known length). 0 = not supplied.",
                    optional=True,
                ),
                IO.Int.Input(
                    "slowdown_factor",
                    default=1,
                    min=1,
                    max=8,
                    tooltip=(
                        "Retime restore. If you slowed the reference clip Nx upstream (so SD tracks fast "
                        "motion more faithfully), set this to N — the OUTPUT frames are sped back up Nx "
                        "(evenly resampled to ~frames/N). 1 = off. Affects the returned `images` frames "
                        "(what you save), not the raw `video` object."
                    ),
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(display_name="video"),
                IO.Image.Output(display_name="images"),
                IO.Image.Output(display_name="last_frame"),
                IO.Float.Output(display_name="output_fps"),
                IO.Int.Output(display_name="output_frames"),
                IO.String.Output(display_name="final_prompt"),
                IO.String.Output(display_name="task_id"),
                IO.String.Output(display_name="api_metadata"),
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        ref_image_asset_urls: str = "",
        ref_image_asset_url: str = "",
        ref_video_asset_url: str = "",
        model: str = "Seedance 2.0 Pro",
        resolution: str = "720p",
        ratio: str = "adaptive",
        duration: int = 5,
        generate_audio: bool = True,
        watermark: bool = False,
        seed: int = -1,
        return_last_frame: bool = True,
        poll_interval_s: float = 9.0,
        poll_timeout_s: float = 1500.0,
        api_key: str = "",
        duration_mode: str = "manual",
        ref_video=None,
        ref_duration_s: float = 0.0,
        slowdown_factor: int = 1,
    ) -> IO.NodeOutput:
        t_start = time.time()

        image_urls = _parse_ref_urls(ref_image_asset_urls, ref_image_asset_url)
        has_video = bool((ref_video_asset_url or "").strip())
        n_images = len(image_urls)

        # Validate inputs BEFORE submit (R0 review) — fail here, not as a late API reject.
        if n_images > _MAX_REF_IMAGES:
            raise ValueError(f"Seedance accepts at most {_MAX_REF_IMAGES} reference images; got {n_images}.")
        # Duration validation is centralized in _resolve_duration (covers all modes).

        final_prompt = (prompt or "").strip()
        if not final_prompt and n_images == 0 and not has_video:
            raise ValueError("No prompt and no refs — can't submit an empty task.")
        prompt_before = final_prompt
        final_prompt = _auto_inject_tags(final_prompt, n_images, has_video)
        tag_injected = final_prompt != prompt_before

        resolved_key = resolve_api_key(api_key, provider="volcengine")
        model_id = _MODELS[model]
        # Measure the LOCAL source clip if wired (works with pre-registered refs);
        # else fall back to a manually-supplied ref_duration_s.
        effective_ref_dur = _measure_video_duration(ref_video) or float(ref_duration_s or 0.0)
        api_duration, duration_note = _resolve_duration(duration_mode, duration, effective_ref_dur)

        print(f"[NV_ByteplusSeedanceGen] model={model_id} res={resolution} ratio={ratio} "
              f"dur={api_duration}s [{duration_note}] slowdown={slowdown_factor} "
              f"refs(img={n_images} vid={'y' if has_video else 'n'}) gen_audio={generate_audio}"
              f"{' (tags auto-injected)' if tag_injected else ''}")

        result = await _generate_one(
            cls=cls,
            final_prompt=final_prompt,
            image_urls=image_urls,
            video_url=ref_video_asset_url,
            model_id=model_id,
            resolution=resolution,
            ratio=ratio,
            api_duration=api_duration,
            generate_audio=generate_audio,
            watermark=watermark,
            seed=seed,
            return_last_frame=return_last_frame,
            slowdown_factor=slowdown_factor,
            poll_interval_s=poll_interval_s,
            poll_timeout_s=poll_timeout_s,
            resolved_key=resolved_key,
        )

        final_resp = result["final_resp"]
        out_fps = result["fps"]
        out_frames = result["frame_count"]
        metadata = {
            "request": {
                "endpoint": _API_BASE,
                "model": model_id,
                "resolution": resolution,
                "ratio": ratio,
                "duration_mode": duration_mode,
                "duration_requested": duration,
                "duration_used": api_duration,
                "duration_source": duration_note,
                "ref_duration_s_effective": round(effective_ref_dur, 3),
                "ref_duration_measured_from_video": ref_video is not None,
                "slowdown_factor": slowdown_factor,
                "retimed": result["retimed"],
                "generate_audio": generate_audio,
                "watermark": watermark,
                "seed": seed,
                "return_last_frame": return_last_frame,
                "n_reference_images": n_images,
                "has_reference_video": has_video,
                "prompt_length": len(final_prompt),
                "prompt_tags_auto_injected": tag_injected,
                "ref_image_kinds": [
                    "asset" if u.startswith("asset://") else ("data" if u.startswith("data:") else "url")
                    for u in image_urls
                ],
            },
            "response": {
                "task_id": result["task_id"],
                "status": final_resp.get("status"),
                "model_echo": final_resp.get("model"),
                "ratio_echo": final_resp.get("ratio"),
                "resolution_echo": final_resp.get("resolution"),
                "duration_echo": final_resp.get("duration"),
                "video_url_tail": _url_path_tail(result["video_url"]),
                "last_frame_url_tail": _url_path_tail(result["last_frame_url"]),
                "last_frame_fetched": result["last_frame_fetched"],
                "raw_frames": result["raw_frame_count"],
                "output_fps": out_fps,
                "output_frames": out_frames,
                "output_duration_s": round(out_frames / out_fps, 3) if out_fps else None,
                "token_usage": _token_usage_summary(final_resp),
            },
            "timing": {
                "total_sec": round(time.time() - t_start, 1),
                "submit_to_done_sec": result["submit_to_done_s"],
            },
        }

        return IO.NodeOutput(
            result["video"],
            result["frames"],
            result["last_frame"],
            out_fps,
            out_frames,
            final_prompt,
            result["task_id"],
            json.dumps(metadata, indent=2),
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_ByteplusSeedanceGen": NV_ByteplusSeedanceGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ByteplusSeedanceGen": "NV BytePlus Seedance Gen",
}
