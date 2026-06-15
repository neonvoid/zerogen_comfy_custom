"""NV Seedance Native Ref Video — direct-to-Volcengine Seedance 2.0 API caller.

Unlike `seedance_ref_fork.py` (which routes through ComfyUI's api.comfy.org
proxy + Comfy org auth), this node calls the Volcengine Ark endpoint directly
using the user's own ark- API key. Useful when:
  - The user has their own Volcengine account + billing.
  - They want access to native-only features that the Comfy proxy doesn't
    expose: audio refs, return_last_frame, web_search tool, service_tier,
    execution_expires_after, safety_identifier.
  - They want to skip the Comfy uploader and pass pre-hosted asset URLs
    directly (or base64 data URIs, or Volcengine asset:// IDs).

Endpoint (from official Volcengine docs, 2026-04):
  - POST https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks
  - GET  https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{id}
  - Bearer auth with ark- prefixed long-lived API key.

API key resolution order: explicit `api_key` input → VOLCENGINE_ARK_API_KEY
env var → ARK_API_KEY env var → .env file.

No pydantic — parses response JSON defensively. The native schema has diverged
from the Comfy proxy mirror in a few fields (e.g. native `status` can be
`expired`; native response may carry `tool_usage`, `last_frame_url`).
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
from .seedance_types import SEEDANCE_UPLOAD_CONFIG

try:
    from comfy.model_management import throw_exception_if_processing_interrupted as _check_interrupt
except ImportError:
    def _check_interrupt() -> None:
        pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
_CREATE_PATH = "/contents/generations/tasks"
_STATUS_PATH = "/contents/generations/tasks"  # + /{id}

# Native model IDs (doubao- prefix — distinct from the dreamina- proxy IDs).
# Pro variant is verified from the official curl example. Fast variant ID
# is inferred from the naming pattern — swap if runtime returns MODEL_NOT_FOUND.
_MODELS = {
    "Seedance 2.0 Pro": "doubao-seedance-2-0-260128",
    "Seedance 2.0 Fast": "doubao-seedance-2-0-fast-260128",
}

_RESOLUTIONS = ["480p", "720p", "1080p"]
_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
_IMAGE_ROLES = ["reference_image", "first_frame", "last_frame", "unused"]

# Pricing mirrors the proxy table for now — update if native pricing differs.
_PRICE_PER_1K_TOKENS = {
    ("doubao-seedance-2-0-260128", False): 0.007,
    ("doubao-seedance-2-0-260128", True): 0.0043,
    ("doubao-seedance-2-0-fast-260128", False): 0.0056,
    ("doubao-seedance-2-0-fast-260128", True): 0.0033,
}


# ---------------------------------------------------------------------------
# Helpers (module-level — keep execute() readable)
# ---------------------------------------------------------------------------

def _auto_inject_tags(prompt: str, n_images: int, has_video: bool, has_audio: bool) -> str:
    """Safety-net tag injection. Mirrors the proxy fork behavior."""
    if not prompt:
        return prompt
    has_image_tag = bool(re.search(r"@Image\s?\d+", prompt))
    has_video_tag = bool(re.search(r"@Video\s?\d+", prompt))
    has_audio_tag = bool(re.search(r"@Audio\s?\d+", prompt))

    parts: list[str] = []
    if has_video and not has_video_tag:
        parts.append("@Video1")
    if has_audio and not has_audio_tag:
        parts.append("@Audio1")
    if n_images > 0 and not has_image_tag:
        parts.extend(f"@Image{i}" for i in range(1, n_images + 1))
    if not parts:
        return prompt
    return f"{' '.join(parts)} {prompt}"


def _analyze_prompt_tags(prompt: str, n_images: int, has_video: bool, has_audio: bool) -> dict:
    image_tags = sorted({int(t) for t in re.findall(r"@Image\s?(\d+)", prompt)})
    video_tags = sorted({int(t) for t in re.findall(r"@Video\s?(\d+)", prompt)})
    audio_tags = sorted({int(t) for t in re.findall(r"@Audio\s?(\d+)", prompt)})
    warnings: list[str] = []
    if image_tags and max(image_tags) > n_images:
        warnings.append(f"prompt references @Image{max(image_tags)} but only {n_images} image(s) in request")
    if video_tags and max(video_tags) > (1 if has_video else 0):
        warnings.append(
            f"prompt references @Video{max(video_tags)} but "
            f"{'only 1 video' if has_video else 'no video'} in request"
        )
    if audio_tags and max(audio_tags) > (1 if has_audio else 0):
        warnings.append(
            f"prompt references @Audio{max(audio_tags)} but "
            f"{'only 1 audio' if has_audio else 'no audio'} in request"
        )
    return {
        "image_tag_indices": image_tags,
        "video_tag_indices": video_tags,
        "audio_tag_indices": audio_tags,
        "warnings": warnings,
    }


def _collect_images(slots: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return non-empty (url, role) image slots, skipping 'unused' role."""
    out = []
    for url, role in slots:
        url = (url or "").strip()
        if not url or role == "unused":
            continue
        out.append((url, role))
    return out


def _validate_image_roles(image_slots: list[tuple[str, str]]) -> None:
    """Enforce the doc's three mutually-exclusive image modes.

    Docs: 图生视频-首帧 | 图生视频-首尾帧 | 多模态参考生视频 (reference_image)
    are three mutually exclusive scenes — roles cannot be mixed across them.
    """
    roles = [r for _, r in image_slots]
    has_ref = "reference_image" in roles
    has_first = "first_frame" in roles
    has_last = "last_frame" in roles

    if has_ref and (has_first or has_last):
        raise ValueError(
            "Seedance image roles: reference_image is mutually exclusive with "
            "first_frame / last_frame. Pick one mode — either all slots use "
            "role=reference_image (multimodal ref), or use first_frame/last_frame "
            "for first/last-frame mode."
        )
    if roles.count("first_frame") > 1:
        raise ValueError("Only one image can have role=first_frame.")
    if roles.count("last_frame") > 1:
        raise ValueError("Only one image can have role=last_frame.")
    if has_last and not has_first:
        raise ValueError(
            "Seedance first/last-frame mode requires a first_frame image when "
            "last_frame is set. Either add a first_frame or change last_frame "
            "to first_frame / reference_image."
        )
    ref_count = roles.count("reference_image")
    if ref_count > 9:
        raise ValueError(f"Seedance 2.0 accepts at most 9 reference images (got {ref_count}).")


def _build_content(prompt: str, image_slots, video_url: str, audio_url: str) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for url, role in image_slots:
        content.append({"type": "image_url", "image_url": {"url": url}, "role": role})
    v = (video_url or "").strip()
    if v:
        content.append({"type": "video_url", "video_url": {"url": v}, "role": "reference_video"})
    a = (audio_url or "").strip()
    if a:
        content.append({"type": "audio_url", "audio_url": {"url": a}, "role": "reference_audio"})
    return content


def _summarize_content(content: list[dict]) -> list[dict]:
    """Shortened content array for logging (no base64 payload dumps)."""
    summary = []
    for i, item in enumerate(content):
        t = item.get("type")
        entry = {"index": i, "type": t, "role": item.get("role", "-")}
        if t == "text":
            text = item.get("text", "")
            entry["text_length"] = len(text)
            entry["text_head"] = text[:120] + ("…" if len(text) > 120 else "")
        else:
            url_obj = item.get(t) or {}
            url = url_obj.get("url", "") if isinstance(url_obj, dict) else ""
            if url.startswith("data:"):
                entry["url_kind"] = f"base64 ({len(url)} chars)"
            elif url.startswith("asset://"):
                entry["url_kind"] = f"asset {url[8:16]}…"
            else:
                entry["url_kind"] = "url"
                entry["url_tail"] = "..." + url[-40:] if len(url) > 40 else url
        summary.append(entry)
    return summary


def _classify_http_error(status_code: int, body_text: str) -> str:
    """Keyword-match response body before suggesting a cause — same pattern as D-059."""
    hints = []
    body_low = body_text.lower()
    if status_code == 401:
        hints.append("401 Unauthorized — check ark- API key value")
    elif status_code == 403:
        hints.append(
            "403 Forbidden — check account balance (≥200 CNY required to unlock "
            "seedance 2.0) or resource package status"
        )
    elif status_code == 429:
        hints.append("429 rate limited — reduce request frequency or raise quota")
    elif status_code == 400:
        if "real person" in body_low or "真人" in body_text:
            hints.append("400 — real-person gate (seedance 2.0 rejects photorealistic human faces in refs)")
        elif "size" in body_low or "too large" in body_low or "request body" in body_low:
            hints.append("400 — payload too large (base64 refs can exceed 64MB body limit; prefer URLs)")
        elif "role" in body_low:
            hints.append("400 — role mismatch (check image role combination is a valid mode)")
        else:
            hints.append("400 — request rejected; see raw message below")
    elif 500 <= status_code < 600:
        hints.append(f"{status_code} transient upstream — retry may help")
    return " | ".join(hints) if hints else f"HTTP {status_code}"


async def _post_task(session: aiohttp.ClientSession, api_key: str, payload: dict) -> dict:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    async with session.post(f"{_API_BASE}{_CREATE_PATH}", json=payload, headers=headers) as resp:
        body_text = await resp.text()
        if resp.status != 200:
            hint = _classify_http_error(resp.status, body_text)
            raise RuntimeError(f"Seedance task creation failed: {hint}\nRaw response:\n{body_text}")
        try:
            return json.loads(body_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Seedance task creation returned non-JSON: {e}\nBody: {body_text[:500]}")


async def _get_task(session: aiohttp.ClientSession, api_key: str, task_id: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{_API_BASE}{_STATUS_PATH}/{task_id}"
    async with session.get(url, headers=headers) as resp:
        body_text = await resp.text()
        if resp.status != 200:
            hint = _classify_http_error(resp.status, body_text)
            raise RuntimeError(f"Seedance task status lookup failed: {hint}\nRaw: {body_text}")
        try:
            return json.loads(body_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Seedance task status returned non-JSON: {e}\nBody: {body_text[:500]}")


async def _poll_task(
    session: aiohttp.ClientSession,
    api_key: str,
    task_id: str,
    interval: float,
    timeout: float,
) -> dict:
    """Poll task status until terminal. Responsive to Comfy Cancel (D-055 pattern)."""
    deadline = time.time() + timeout
    chunked_sleep_step = 0.5  # chunk sleeps so cancel fires within 0.5s

    last_status = None
    while True:
        _check_interrupt()
        resp = await _get_task(session, api_key, task_id)
        status = resp.get("status")
        if status != last_status:
            print(f"[NV_SeedanceNative] status: {status}")
            last_status = status
        if status in ("succeeded", "failed", "expired", "cancelled"):
            return resp
        if time.time() > deadline:
            raise RuntimeError(
                f"Seedance task poll timed out after {timeout:.0f}s (last status: {status}). "
                f"Task id: {task_id}. Status-poll deadline tunable via `poll_timeout_s`."
            )
        elapsed = 0.0
        while elapsed < interval:
            _check_interrupt()
            step = min(chunked_sleep_step, interval - elapsed)
            await asyncio.sleep(step)
            elapsed += step


def _token_usage_summary(resp: dict, model_id: str, has_video: bool) -> dict:
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

class NV_SeedanceNativeRefVideo(IO.ComfyNode):
    """Seedance 2.0 via direct-to-Volcengine API (user-supplied ark- key).

    Takes pre-hosted asset URLs (HTTPS, base64 data URIs, or asset:// IDs)
    and submits them to the native Volcengine endpoint. No ComfyUI proxy
    auth, no asset uploader — you bring the URLs, you bring the key.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_SeedanceNativeRefVideo",
            display_name="NV Seedance Native Ref Video",
            category="NV_Utils/api",
            description=(
                "Direct-to-Volcengine Seedance 2.0 ref-to-video caller. Uses your own "
                "ark- API key (env: VOLCENGINE_ARK_API_KEY / ARK_API_KEY, or .env). "
                "Accepts pre-hosted URLs, base64 data URIs, or asset:// IDs for refs."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Text prompt describing the video. CN ≤500 chars, EN ≤1000 words. "
                        "Weave @Image1..N / @Video1 / @Audio1 tags inline for adherence."
                    ),
                ),
                SEEDANCE_UPLOAD_CONFIG.Input(
                    "upload_config",
                    tooltip=(
                        "Optional — wire the `config` output of NV_SeedancePrep here to use "
                        "tensor-based refs (IMAGE / VIDEO) uploaded via ComfyUI's asset host. "
                        "When provided, overrides the image_N_url / video_url string slots "
                        "below. Images default to role=reference_image (multimodal mode). "
                        "For first_frame/last_frame mode, use the URL slots instead. Leave "
                        "unwired if you're pasting URLs directly."
                    ),
                    optional=True,
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
                    tooltip="480p / 720p / 1080p. Fast variant does NOT support 1080p.",
                ),
                IO.Combo.Input(
                    "ratio",
                    options=_RATIOS,
                    default="adaptive",
                    tooltip="adaptive = model picks based on refs. Docs: 16:9,4:3,1:1,3:4,9:16,21:9.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=-1,
                    max=15,
                    step=1,
                    tooltip="Seconds. 4-15, or -1 = auto (model picks). `frames` is not supported on 2.0.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=True,
                    tooltip="Produce synchronized audio (voice/SFX/BGM) matching prompt + visuals.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Add the ByteDance / Volcengine watermark to the output.",
                ),
                IO.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="-1 = random. Non-deterministic even at fixed seed (best-effort similarity).",
                ),
                IO.Boolean.Input(
                    "return_last_frame",
                    default=False,
                    tooltip="If True, Volcengine returns the output video's last frame as a PNG URL (surfaced in api_metadata.last_frame_url).",
                ),
                IO.Boolean.Input(
                    "web_search",
                    default=False,
                    tooltip="Enable the web_search tool (seedance 2.0 only). Model decides per-prompt whether to search.",
                ),
                IO.Int.Input(
                    "execution_expires_after",
                    default=172800,
                    min=3600,
                    max=259200,
                    step=3600,
                    tooltip="Seconds before task auto-expires. Default 48h. Range: 1h-72h.",
                ),
                IO.String.Input(
                    "safety_identifier",
                    default="",
                    tooltip=(
                        "Optional hashed per-user ID (≤64 chars) for Volcengine abuse detection. "
                        "Leave empty if not required."
                    ),
                    optional=True,
                ),
                IO.String.Input(
                    "api_key",
                    default="",
                    tooltip=(
                        "Optional override. Empty → env (VOLCENGINE_ARK_API_KEY / ARK_API_KEY) → .env."
                    ),
                    optional=True,
                ),
                # Reference image slots (up to 4 — docs allow 9, but 4 covers the common cases).
                IO.String.Input(
                    "image_1_url",
                    default="",
                    tooltip="HTTPS URL, data:image/...;base64,... , or asset://<ID>. Empty = skip.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "image_1_role",
                    options=_IMAGE_ROLES,
                    default="reference_image",
                    tooltip="reference_image (multimodal), first_frame, last_frame. 'unused' = skip slot.",
                ),
                IO.String.Input("image_2_url", default="", optional=True),
                IO.Combo.Input("image_2_role", options=_IMAGE_ROLES, default="reference_image"),
                IO.String.Input("image_3_url", default="", optional=True),
                IO.Combo.Input("image_3_role", options=_IMAGE_ROLES, default="reference_image"),
                IO.String.Input("image_4_url", default="", optional=True),
                IO.Combo.Input("image_4_role", options=_IMAGE_ROLES, default="reference_image"),
                IO.String.Input(
                    "video_url",
                    default="",
                    tooltip="Optional reference_video URL. mp4/mov, 2-15s, 480-1080p, ≤50MB.",
                    optional=True,
                ),
                IO.String.Input(
                    "audio_url",
                    default="",
                    tooltip="Optional reference_audio URL. wav/mp3, 2-15s, ≤15MB. Requires ≥1 image or video.",
                    optional=True,
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
                    default=600.0,
                    min=60.0,
                    max=1800.0,
                    step=30.0,
                    tooltip="Max seconds to wait for task completion before giving up.",
                ),
                IO.String.Input(
                    "callback_url",
                    default="",
                    tooltip="Optional HTTPS callback for status updates. Leave empty for polling-only.",
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
        prompt: str,
        upload_config: dict | None,
        model: str,
        resolution: str,
        ratio: str,
        duration: int,
        generate_audio: bool,
        watermark: bool,
        seed: int,
        return_last_frame: bool,
        web_search: bool,
        execution_expires_after: int,
        safety_identifier: str = "",
        api_key: str = "",
        image_1_url: str = "", image_1_role: str = "reference_image",
        image_2_url: str = "", image_2_role: str = "reference_image",
        image_3_url: str = "", image_3_role: str = "reference_image",
        image_4_url: str = "", image_4_role: str = "reference_image",
        video_url: str = "",
        audio_url: str = "",
        poll_interval_s: float = 9.0,
        poll_timeout_s: float = 600.0,
        callback_url: str = "",
    ) -> IO.NodeOutput:
        t_start = time.time()

        prompt = (prompt or "").strip()
        if not prompt and not any([image_1_url, image_2_url, image_3_url, image_4_url, video_url]):
            raise ValueError(
                "Seedance requires prompt OR at least one reference (image/video). Audio "
                "alone is rejected by the API."
            )

        resolved_key = resolve_api_key(api_key, provider="volcengine")
        model_id = _MODELS[model]

        # --- source refs: upload_config path wins over URL slots ---
        ref_source = "url_slots"
        if upload_config is not None and isinstance(upload_config, dict):
            cfg_images = upload_config.get("uploaded_image_urls") or []
            cfg_video = upload_config.get("uploaded_video_url")
            if cfg_images or cfg_video:
                ref_source = "upload_config"
                # All config-sourced images default to reference_image (multimodal mode).
                image_slots = [(u, "reference_image") for u in cfg_images]
                video_url = cfg_video or ""
                # Audio stays URL-only — seedance_prep doesn't upload audio.
                print(
                    f"[NV_SeedanceNative] Using upload_config: {len(cfg_images)} image(s), "
                    f"video={'yes' if cfg_video else 'no'} (URL slots ignored)"
                )

        if ref_source == "url_slots":
            raw_slots = [
                (image_1_url, image_1_role),
                (image_2_url, image_2_role),
                (image_3_url, image_3_role),
                (image_4_url, image_4_role),
            ]
            image_slots = _collect_images(raw_slots)

        _validate_image_roles(image_slots)

        has_video = bool((video_url or "").strip())
        has_audio = bool((audio_url or "").strip())
        n_images = len(image_slots)

        if has_audio and not (n_images > 0 or has_video):
            raise ValueError("Seedance rejects audio-only refs — include at least 1 image or video.")

        if n_images == 0 and not has_video:
            # Pure text-to-video is allowed — no extra check needed here.
            pass

        # --- safety-net tag injection + diagnostics ---
        prompt_before = prompt
        prompt = _auto_inject_tags(prompt, n_images, has_video, has_audio)
        tag_injected = prompt != prompt_before
        tag_analysis = _analyze_prompt_tags(prompt, n_images, has_video, has_audio)

        # --- build content + top-level payload ---
        content = _build_content(prompt, image_slots, video_url, audio_url)
        payload: dict = {
            "model": model_id,
            "content": content,
            "resolution": resolution,
            "ratio": ratio,
            "duration": duration,
            "generate_audio": generate_audio,
            "watermark": watermark,
            "return_last_frame": return_last_frame,
            "execution_expires_after": execution_expires_after,
        }
        if seed != -1:
            payload["seed"] = seed
        if web_search:
            payload["tools"] = [{"type": "web_search"}]
        if safety_identifier.strip():
            payload["safety_identifier"] = safety_identifier.strip()
        if callback_url.strip():
            payload["callback_url"] = callback_url.strip()

        content_summary = _summarize_content(content)
        print(f"[NV_SeedanceNative] Model: {model_id} | res={resolution} ratio={ratio} dur={duration}s")
        print(f"[NV_SeedanceNative] Refs: images={n_images} video={'y' if has_video else 'n'} audio={'y' if has_audio else 'n'}")
        print(f"[NV_SeedanceNative] seed={seed} gen_audio={generate_audio} watermark={watermark} "
              f"return_last_frame={return_last_frame} web_search={web_search}")
        print(f"[NV_SeedanceNative] Tag analysis: "
              f"@Image{tag_analysis['image_tag_indices'] or '[]'} "
              f"@Video{tag_analysis['video_tag_indices'] or '[]'} "
              f"@Audio{tag_analysis['audio_tag_indices'] or '[]'}"
              f"{' (auto-injected)' if tag_injected else ''}")
        for w in tag_analysis["warnings"]:
            print(f"[NV_SeedanceNative] ⚠ {w}")
        print(f"[NV_SeedanceNative] Content array ({len(content)} items):")
        for e in content_summary:
            line = f"  [{e['index']}] {e['type']} role={e.get('role', '-')}"
            if "text_head" in e:
                line += f" text={e['text_head']!r} ({e['text_length']} chars)"
            elif "url_tail" in e:
                line += f" url={e['url_tail']}"
            elif "url_kind" in e:
                line += f" {e['url_kind']}"
            print(line)

        t_submit = time.time()

        # Windows asyncio (ProactorEventLoop) + aiohttp's default session can
        # leak zombie SSL contexts when a request fails mid-flight (observed
        # after a 400 real-person-gate rejection: ComfyUI UI froze post-error).
        # Explicit timeouts + force_close connector + try/finally teardown
        # (instead of `async with`) fix it defensively.
        session_timeout = aiohttp.ClientTimeout(
            total=None,         # no total cap — we have our own poll_timeout
            connect=30,
            sock_connect=30,
            sock_read=120,
        )
        connector = aiohttp.TCPConnector(force_close=True, limit=8)
        session = aiohttp.ClientSession(timeout=session_timeout, connector=connector)
        try:
            create_resp = await _post_task(session, resolved_key, payload)
            task_id = create_resp.get("id")
            if not task_id:
                raise RuntimeError(f"Seedance task creation returned no id. Raw: {create_resp}")
            print(f"[NV_SeedanceNative] Task submitted: {task_id}")

            final_resp = await _poll_task(session, resolved_key, task_id, poll_interval_s, poll_timeout_s)
        finally:
            await session.close()
            # Give the event loop a tick so Windows' ProactorEventLoop finishes
            # closing SSL contexts before the exception (if any) unwinds further.
            await asyncio.sleep(0.1)

        t_done = time.time()

        status = final_resp.get("status")
        if status != "succeeded":
            err = final_resp.get("error") or {}
            raise RuntimeError(
                f"Seedance task ended with status={status}. "
                f"error.code={err.get('code')!r} error.message={err.get('message')!r}. "
                f"Task id: {task_id}"
            )

        resp_content = final_resp.get("content") or {}
        result_video_url = resp_content.get("video_url")
        last_frame_url = resp_content.get("last_frame_url")
        if not result_video_url:
            raise RuntimeError(
                f"Seedance task succeeded but content.video_url is missing. Raw: {final_resp}"
            )

        output_video = await download_url_to_video_output(result_video_url)
        try:
            components = output_video.get_components()
            out_images = components.images
            out_fps = float(components.frame_rate)
            out_frames = int(out_images.shape[0])
        except Exception as e:
            print(f"[NV_SeedanceNative] Warning: frame decode failed: {e}")
            out_images = torch.zeros(1, 64, 64, 3)
            out_fps = 0.0
            out_frames = 0

        # --- optional: fetch the returned last-frame PNG as an IMAGE tensor ---
        # Soft-fails to a 64x64 zero tensor so a missing PNG doesn't kill an
        # otherwise-successful task. Typical use: wire last_frame into the next
        # chunk's first_frame input for continuous multi-shot generation.
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
                print(f"[NV_SeedanceNative] last_frame fetched: {arr.shape[1]}x{arr.shape[0]} (HxW inverted) -> tensor {tuple(last_frame_tensor.shape)}")
            except Exception as e:
                print(f"[NV_SeedanceNative] Warning: last_frame fetch failed: {e}")
        elif return_last_frame and not last_frame_url:
            print("[NV_SeedanceNative] Note: return_last_frame=True but API did not return last_frame_url")

        t_end = time.time()

        token_usage = _token_usage_summary(final_resp, model_id, has_video)
        if token_usage["cost_estimate_usd"] is not None:
            print(f"[NV_SeedanceNative] Tokens: total={token_usage['total_tokens']} "
                  f"cost≈${token_usage['cost_estimate_usd']}")

        metadata = {
            "request": {
                "model": model_id,
                "resolution": resolution,
                "ratio": ratio,
                "duration": duration,
                "generate_audio": generate_audio,
                "watermark": watermark,
                "seed": seed,
                "return_last_frame": return_last_frame,
                "web_search": web_search,
                "execution_expires_after": execution_expires_after,
                "has_callback": bool(callback_url.strip()),
                "has_safety_identifier": bool(safety_identifier.strip()),
                "ref_source": ref_source,
                "n_reference_images": n_images,
                "image_roles": [r for _, r in image_slots],
                "has_reference_video": has_video,
                "has_reference_audio": has_audio,
                "prompt_length": len(prompt),
                "prompt_tags_auto_injected": tag_injected,
                "tag_analysis": tag_analysis,
                "content_array": content_summary,
            },
            "response": {
                "task_id": task_id,
                "status": status,
                "model_echo": final_resp.get("model"),
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
            prompt,
            task_id,
            json.dumps(metadata, indent=2),
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_SeedanceNativeRefVideo": NV_SeedanceNativeRefVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SeedanceNativeRefVideo": "NV Seedance Native Ref Video",
}
