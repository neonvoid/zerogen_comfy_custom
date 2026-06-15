"""NV Seedance Native Ref Video V2 — config-only, slim caller.

Differences from V1 (nv_seedance_native.py):
  - Consumes ONLY SEEDANCE_UPLOAD_CONFIG_V2 (no URL slots, no role dropdowns)
  - Role assignments come from the config (built in Prep V2), not user widgets
  - ~10 generation widgets instead of 25+
  - Same polling, auth, error classification, session hardening as V1
  - Same outputs including decoded last_frame IMAGE tensor

If you need URL-based refs, use the legacy NV_SeedanceNativeRefVideo.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from io import BytesIO

import aiohttp
import numpy as np
import torch
from PIL import Image

from comfy_api.latest import IO
from comfy_api_nodes.util import download_url_to_video_output
from comfy_api_nodes.util.download_helpers import download_url_to_bytesio

from .api_keys import resolve_api_key
from .nv_seedance_upload_utils import (
    MODE_BRIDGE,
    MODE_FIRST_FRAME,
    MODE_MULTIMODAL,
    MODE_TEXT_ONLY,
    SEEDANCE_UPLOAD_CONFIG_V2,
)

try:
    from comfy.model_management import throw_exception_if_processing_interrupted as _check_interrupt
except ImportError:
    def _check_interrupt() -> None:
        pass


# ---------------------------------------------------------------------------
# Constants — same as V1 (don't duplicate, but V1 may change independently)
# ---------------------------------------------------------------------------

_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
_CREATE_PATH = "/contents/generations/tasks"
_STATUS_PATH = "/contents/generations/tasks"

_MODELS = {
    "Seedance 2.0 Pro": "doubao-seedance-2-0-260128",
    "Seedance 2.0 Fast": "doubao-seedance-2-0-fast-260128",
}

_RESOLUTIONS = ["480p", "720p", "1080p"]
_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]

_PRICE_PER_1K_TOKENS = {
    ("doubao-seedance-2-0-260128", False): 0.007,
    ("doubao-seedance-2-0-260128", True): 0.0043,
    ("doubao-seedance-2-0-fast-260128", False): 0.0056,
    ("doubao-seedance-2-0-fast-260128", True): 0.0033,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_inject_tags(prompt: str, n_images: int, has_video: bool) -> str:
    """Mirror V1 auto-injection: @Image1..N @Video1 prefix if absent.

    Detects BOTH English `@Image1`/`@Video1` AND Chinese `[图1]`/`[视频1]`/`[音频1]`
    bracket tags. If either form is present in the prompt, no auto-injection.
    Prevents duplicate-tag pollution when the prompt has been translated to CN.

    When auto-injecting (prompt has neither tag form), uses Chinese brackets if the
    prompt contains any CJK characters (heuristic for "this is a CN-prompt context");
    otherwise uses English @ tags.
    """
    if not prompt:
        return prompt

    has_image_tag = bool(re.search(r"@Image\s?\d+", prompt)) or bool(re.search(r"\[图\s?\d+\]", prompt))
    has_video_tag = bool(re.search(r"@Video\s?\d+", prompt)) or bool(re.search(r"\[视频\s?\d+\]", prompt))

    # Heuristic: prompt contains CJK ideographs → use CN bracket form for any injection
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", prompt))

    parts: list[str] = []
    if has_video and not has_video_tag:
        parts.append("[视频1]" if has_cjk else "@Video1")
    if n_images > 0 and not has_image_tag:
        if has_cjk:
            parts.extend(f"[图{i}]" for i in range(1, n_images + 1))
        else:
            parts.extend(f"@Image{i}" for i in range(1, n_images + 1))
    if not parts:
        return prompt
    return f"{' '.join(parts)} {prompt}"


def _classify_http_error(status_code: int, body_text: str) -> str:
    hints = []
    body_low = body_text.lower()
    if status_code == 401:
        hints.append("401 Unauthorized — check ark- API key value")
    elif status_code == 403:
        hints.append(
            "403 Forbidden — check account balance (≥200 CNY required to unlock seedance 2.0) "
            "or resource package status"
        )
    elif status_code == 429:
        hints.append("429 rate limited — reduce request frequency or raise quota")
    elif status_code == 400:
        if "real person" in body_low or "真人" in body_text:
            hints.append("400 — real-person gate (seedance 2.0 rejects photorealistic human faces in refs)")
        elif "size" in body_low or "too large" in body_low:
            hints.append("400 — payload too large (rare with URL-based refs; check ref sizes)")
        elif "role" in body_low:
            hints.append("400 — role mismatch (Prep V2 should have caught this; upstream bug?)")
        else:
            hints.append("400 — request rejected; see raw message below")
    elif 500 <= status_code < 600:
        hints.append(f"{status_code} transient upstream — retry may help")
    return " | ".join(hints) if hints else f"HTTP {status_code}"


def _build_api_content(config: dict, final_prompt: str, ref_priority: str = "as_config") -> list[dict]:
    """Convert v2 SEEDANCE_UPLOAD_CONFIG_V2 → API-ready `content` array.

    `ref_priority` controls the order of image vs video items in the content
    array. The hypothesis (Codex multi-AI 2026-04-23) is that earlier-positioned
    refs may receive higher attention weight in the model's vision encoder.

      - `as_config`:    keep config order (default — image then video)
      - `image_first`:  explicitly put image refs before video refs
      - `video_first`:  put video refs before image refs (test for image-dominance fix)
    """
    items = config.get("content", [])
    images = [c for c in items if c.get("kind") == "image"]
    videos = [c for c in items if c.get("kind") == "video"]

    if ref_priority == "image_first":
        ordered = images + videos
    elif ref_priority == "video_first":
        ordered = videos + images
    else:  # as_config
        ordered = items

    content: list[dict] = [{"type": "text", "text": final_prompt}]
    for item in ordered:
        kind = item.get("kind")
        if kind == "image":
            content.append({
                "type": "image_url",
                "image_url": {"url": item["url"]},
                "role": item["role"],
            })
        elif kind == "video":
            content.append({
                "type": "video_url",
                "video_url": {"url": item["url"]},
                "role": item["role"],
            })
    return content


async def _post_task(session, api_key, payload):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    async with session.post(f"{_API_BASE}{_CREATE_PATH}", json=payload, headers=headers) as resp:
        body_text = await resp.text()
        if resp.status != 200:
            hint = _classify_http_error(resp.status, body_text)
            raise RuntimeError(f"Seedance task creation failed: {hint}\nRaw response:\n{body_text}")
        try:
            return json.loads(body_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Seedance task creation non-JSON: {e}\nBody: {body_text[:500]}")


async def _get_task(session, api_key, task_id):
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{_API_BASE}{_STATUS_PATH}/{task_id}"
    async with session.get(url, headers=headers) as resp:
        body_text = await resp.text()
        if resp.status != 200:
            hint = _classify_http_error(resp.status, body_text)
            raise RuntimeError(f"Seedance status lookup failed: {hint}\nRaw: {body_text}")
        try:
            return json.loads(body_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Seedance status non-JSON: {e}\nBody: {body_text[:500]}")


# Transient network failures during long polls (Beijing endpoint, 5-75min jobs).
# A single TLS handshake reset or DNS hiccup must NOT forfeit the whole task —
# Volcengine's server-side job continues regardless of our polling client.
_TRANSIENT_POLL_ERRORS = (
    aiohttp.ClientConnectionError,    # parent of ClientConnectorError, ServerDisconnectedError
    aiohttp.ClientPayloadError,
    asyncio.TimeoutError,
    ConnectionResetError,
    ConnectionError,
)
_MAX_CONSECUTIVE_POLL_FAILURES = 10   # ~5 min of unreachable at default interval = real outage
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
                    f"Seedance poll failed {consecutive_failures}× consecutively — "
                    f"likely a real outage. task_id={task_id}. Last error: {type(e).__name__}: {e}. "
                    f"Re-run with NV_SeedanceFetchTask once connectivity recovers."
                ) from e
            backoff = min(_POLL_RETRY_BACKOFF_CAP_S, 2.0 * consecutive_failures)
            print(f"[NV_SeedanceNative_V2] poll #{consecutive_failures} transient error "
                  f"({type(e).__name__}: {e}); retrying in {backoff:.1f}s. task_id={task_id}")
            elapsed = 0.0
            while elapsed < backoff:
                _check_interrupt()
                step = min(0.5, backoff - elapsed)
                await asyncio.sleep(step)
                elapsed += step
            # Don't count retry sleeps against the wall-clock deadline; transient
            # outages should consume retry budget, not job-completion budget.
            deadline += backoff
            continue
        consecutive_failures = 0  # reset on any successful GET
        status = resp.get("status")
        if status != last_status:
            print(f"[NV_SeedanceNative_V2] status: {status}")
            last_status = status
        if status in ("succeeded", "failed", "expired", "cancelled"):
            return resp
        if time.time() > deadline:
            raise RuntimeError(
                f"Seedance task poll timed out after {timeout:.0f}s (last: {status}). task_id={task_id}"
            )
        elapsed = 0.0
        while elapsed < interval:
            _check_interrupt()
            step = min(0.5, interval - elapsed)
            await asyncio.sleep(step)
            elapsed += step


def _token_usage_summary(resp, model_id, has_video):
    usage = resp.get("usage") or {}
    out = {
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "tool_usage": usage.get("tool_usage"),
        "cost_estimate_usd": None,
        "cost_formula": None,
    }
    total = out["total_tokens"]
    if isinstance(total, (int, float)):
        rate = _PRICE_PER_1K_TOKENS.get((model_id, has_video))
        if rate is not None:
            cost = total * 1.43 * rate / 1000.0
            out["cost_estimate_usd"] = round(cost, 4)
            out["cost_formula"] = f"{total} × 1.43 × ${rate}/1K = ${cost:.4f}"
    return out


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class NV_SeedanceNativeRefVideo_V2(IO.ComfyNode):
    """Seedance 2.0 native API caller. Config-only input.

    Consumes SEEDANCE_UPLOAD_CONFIG_V2 from NV Seedance Prep V2. Role
    assignments and the 3-mode mutual exclusion rule are already enforced
    upstream — this node trusts the config and submits.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_SeedanceNativeRefVideo_V2",
            display_name="NV Seedance Native Ref Video V2",
            category="NV_Utils/api",
            description=(
                "Seedance 2.0 native API caller (config-only). Pair with NV Seedance Prep V2 "
                "for tensor-in workflows. Uses your ark- key from env VOLCENGINE_ARK_API_KEY / "
                "ARK_API_KEY or .env file."
            ),
            inputs=[
                SEEDANCE_UPLOAD_CONFIG_V2.Input(
                    "upload_config",
                    tooltip="Wire from NV Seedance Prep V2's upload_config output.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Final prompt sent to the API. If empty, uses the draft prompt embedded "
                        "in upload_config. Overrides it if non-empty. @Image/@Video tags auto-injected "
                        "if absent."
                    ),
                ),
                IO.Combo.Input(
                    "model",
                    options=list(_MODELS.keys()),
                    default="Seedance 2.0 Pro",
                    tooltip="Pro = quality, Fast = ~20% cheaper (no 1080p on Fast).",
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
                    tooltip="adaptive = model picks based on refs.",
                ),
                IO.Combo.Input(
                    "duration_mode",
                    options=["manual", "auto_from_ref_video"],
                    default="auto_from_ref_video",
                    tooltip=(
                        "'auto_from_ref_video' = match ref video duration (ceil, clamped 4-15s). "
                        "Falls back to the duration slider if upload_config has no ref video. "
                        "'manual' = always use the slider below."
                    ),
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=-1,
                    max=15,
                    step=1,
                    tooltip="Output seconds. 4-15 or -1 for model-auto. Used when duration_mode='manual' or no ref video.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=True,
                    tooltip=(
                        "Produce synchronized audio. Gemini hypothesis: disable during face-swap "
                        "to prevent lip-sync phoneme-driven face warping (unverified but free to test)."
                    ),
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Add Volcengine watermark to output.",
                ),
                IO.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="-1 = random. Non-deterministic even at fixed seed.",
                ),
                IO.Boolean.Input(
                    "return_last_frame",
                    default=True,
                    tooltip=(
                        "Return the output's last frame as PNG — output wires directly into another "
                        "Prep V2's first_frame slot for chunk chaining."
                    ),
                ),
                IO.Boolean.Input(
                    "web_search",
                    default=False,
                    tooltip="Enable web_search tool — model decides per-prompt.",
                ),
                IO.Combo.Input(
                    "ref_priority",
                    options=["as_config", "image_first", "video_first"],
                    default="as_config",
                    tooltip=(
                        "Order of refs in the API content array. as_config = default Prep order. "
                        "video_first = put video refs before image refs (test fix for Mode C "
                        "image-dominance issue — earlier-positioned refs may attention-weigh higher)."
                    ),
                ),
                IO.Float.Input(
                    "poll_interval_s",
                    default=9.0,
                    min=2.0,
                    max=60.0,
                    step=1.0,
                    tooltip="Seconds between status polls.",
                ),
                IO.Float.Input(
                    "poll_timeout_s",
                    default=1500.0,
                    min=60.0,
                    max=7200.0,
                    step=30.0,
                    tooltip=(
                        "Max seconds to wait for task completion. Benchmarks: 5s Pro 720p ≈ 5min, "
                        "15s Mode C Pro with 5 refs + ref_video ≈ 75min (measured 2026-04-24). "
                        "On timeout, task_id is logged — use NV_SeedanceFetchTask later to retrieve "
                        "the finished video without re-running the generation."
                    ),
                ),
                IO.String.Input(
                    "api_key",
                    default="",
                    tooltip="Optional override. Empty → env / .env.",
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
        )

    @classmethod
    async def execute(
        cls,
        upload_config: dict,
        prompt: str,
        model: str,
        resolution: str,
        ratio: str,
        duration_mode: str,
        duration: int,
        generate_audio: bool,
        watermark: bool,
        seed: int,
        return_last_frame: bool,
        web_search: bool,
        ref_priority: str,
        poll_interval_s: float,
        poll_timeout_s: float,
        api_key: str = "",
    ) -> IO.NodeOutput:
        t_start = time.time()

        if not isinstance(upload_config, dict) or upload_config.get("schema_version") != 2:
            raise ValueError(
                "NV Seedance Native Ref Video V2 requires a SEEDANCE_UPLOAD_CONFIG_V2 from "
                "NV Seedance Prep V2. Did you wire a legacy config?"
            )

        mode = upload_config.get("mode", MODE_TEXT_ONLY)
        content_items = upload_config.get("content", [])
        n_images = sum(1 for c in content_items if c.get("kind") == "image")
        has_video = any(c.get("kind") == "video" for c in content_items)

        # Prompt resolution: explicit input > config-embedded > error
        final_prompt = (prompt or "").strip() or (upload_config.get("prompt") or "").strip()
        if not final_prompt and mode == MODE_TEXT_ONLY and n_images == 0 and not has_video:
            raise ValueError("No prompt and no refs — can't submit an empty task.")

        # Safety-net tag injection
        prompt_before = final_prompt
        final_prompt = _auto_inject_tags(final_prompt, n_images, has_video)
        tag_injected = final_prompt != prompt_before

        resolved_key = resolve_api_key(api_key, provider="volcengine")
        model_id = _MODELS[model]

        # --- resolve duration ---
        import math
        ref_dur = (upload_config.get("provenance") or {}).get("ref_video_duration_s")
        if duration_mode == "auto_from_ref_video" and ref_dur:
            api_duration = max(4, min(15, math.ceil(float(ref_dur))))
            duration_source = f"auto (from ref video {ref_dur:.2f}s → ceil → {api_duration}s, clamped 4-15)"
        else:
            api_duration = duration
            if duration_mode == "auto_from_ref_video":
                duration_source = f"auto-fallback to manual ({duration}s) — no ref video duration in config"
            else:
                duration_source = f"manual ({duration}s)"
        print(f"[NV_SeedanceNative_V2] Duration: {duration_source}")

        api_content = _build_api_content(upload_config, final_prompt, ref_priority=ref_priority)

        payload: dict = {
            "model": model_id,
            "content": api_content,
            "resolution": resolution,
            "ratio": ratio,
            "duration": api_duration,
            "generate_audio": generate_audio,
            "watermark": watermark,
            "return_last_frame": return_last_frame,
        }
        if seed != -1:
            payload["seed"] = seed
        if web_search:
            payload["tools"] = [{"type": "web_search"}]

        print(f"[NV_SeedanceNative_V2] Mode: {mode} | Model: {model_id} | res={resolution} "
              f"ratio={ratio} dur={api_duration}s")
        print(f"[NV_SeedanceNative_V2] Refs: images={n_images} video={'y' if has_video else 'n'}")
        print(f"[NV_SeedanceNative_V2] seed={seed} gen_audio={generate_audio} "
              f"return_last_frame={return_last_frame} web_search={web_search}"
              f"{' (tags auto-injected)' if tag_injected else ''}")

        t_submit = time.time()

        session_timeout = aiohttp.ClientTimeout(
            total=None, connect=30, sock_connect=30, sock_read=120,
        )
        connector = aiohttp.TCPConnector(force_close=True, limit=8)
        session = aiohttp.ClientSession(timeout=session_timeout, connector=connector)
        try:
            create_resp = await _post_task(session, resolved_key, payload)
            task_id = create_resp.get("id")
            if not task_id:
                raise RuntimeError(f"Task creation returned no id. Raw: {create_resp}")
            print(f"[NV_SeedanceNative_V2] Task submitted: {task_id}")
            final_resp = await _poll_task(session, resolved_key, task_id, poll_interval_s, poll_timeout_s)
        finally:
            await session.close()
            await asyncio.sleep(0.1)  # Windows ProactorEventLoop SSL cleanup tick

        t_done = time.time()

        status = final_resp.get("status")
        if status != "succeeded":
            err = final_resp.get("error") or {}
            raise RuntimeError(
                f"Seedance task ended status={status}. "
                f"error.code={err.get('code')!r} error.message={err.get('message')!r}. "
                f"task_id={task_id}"
            )

        resp_content = final_resp.get("content") or {}
        result_video_url = resp_content.get("video_url")
        last_frame_url = resp_content.get("last_frame_url")
        if not result_video_url:
            raise RuntimeError(f"Task succeeded but content.video_url missing. Raw: {final_resp}")

        output_video = await download_url_to_video_output(result_video_url)
        try:
            components = output_video.get_components()
            out_images = components.images
            out_fps = float(components.frame_rate)
            out_frames = int(out_images.shape[0])
        except Exception as e:
            print(f"[NV_SeedanceNative_V2] Warning: frame decode failed: {e}")
            out_images = torch.zeros(1, 64, 64, 3)
            out_fps = 0.0
            out_frames = 0

        # Optional: fetch the last_frame PNG
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
                print(f"[NV_SeedanceNative_V2] last_frame fetched: {tuple(last_frame_tensor.shape)}")
            except Exception as e:
                print(f"[NV_SeedanceNative_V2] Warning: last_frame fetch failed: {e}")

        t_end = time.time()

        token_usage = _token_usage_summary(final_resp, model_id, has_video)
        if token_usage["cost_estimate_usd"] is not None:
            print(f"[NV_SeedanceNative_V2] Tokens: total={token_usage['total_tokens']} "
                  f"cost≈${token_usage['cost_estimate_usd']}")

        metadata = {
            "request": {
                "mode": mode,
                "model": model_id,
                "resolution": resolution,
                "ratio": ratio,
                "duration_mode": duration_mode,
                "duration_requested": duration,
                "duration_used": api_duration,
                "duration_source": duration_source,
                "generate_audio": generate_audio,
                "watermark": watermark,
                "seed": seed,
                "return_last_frame": return_last_frame,
                "web_search": web_search,
                "ref_priority": ref_priority,
                "n_reference_images": n_images,
                "has_reference_video": has_video,
                "prompt_length": len(final_prompt),
                "prompt_tags_auto_injected": tag_injected,
                "upload_config_provenance": upload_config.get("provenance", {}),
            },
            "response": {
                "task_id": task_id,
                "status": status,
                "model_echo": final_resp.get("model"),
                "ratio_echo": final_resp.get("ratio"),
                "resolution_echo": final_resp.get("resolution"),
                "duration_echo": final_resp.get("duration"),
                "video_url_tail": result_video_url[-60:] if result_video_url else None,
                "last_frame_url": last_frame_url,
                "last_frame_fetched": last_frame_fetched,
                "output_fps": out_fps,
                "output_frames": out_frames,
                "output_duration_s": round(out_frames / out_fps, 3) if out_fps else None,
                "top_level_keys": sorted(final_resp.keys()),
                "content_keys": sorted(resp_content.keys()) if isinstance(resp_content, dict) else None,
                "token_usage": token_usage,
            },
            "timing": {
                "submit_sec": round(t_submit - t_start, 1),
                "processing_sec": round(t_done - t_submit, 1),
                "download_sec": round(t_end - t_done, 1),
                "total_sec": round(t_end - t_start, 1),
            },
        }

        return IO.NodeOutput(
            output_video,
            out_images,
            last_frame_tensor,
            out_fps,
            out_frames,
            final_prompt,
            task_id,
            json.dumps(metadata, indent=2),
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_SeedanceNativeRefVideo_V2": NV_SeedanceNativeRefVideo_V2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SeedanceNativeRefVideo_V2": "NV Seedance Native Ref Video V2",
}
