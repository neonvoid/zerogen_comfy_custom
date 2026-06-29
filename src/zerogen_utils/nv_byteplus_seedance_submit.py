"""BytePlus Seedance Submit — non-blocking fire-and-forget gen.

The submit half of the submit/watch split. Builds the SAME payload as
Zerogen_ByteplusSeedanceGen (reusing its validated helpers) but stops after the
POST: it returns the task_id in ~2s and frees ComfyUI's execution slot instead of
blocking ~5 min on the poll.

On submit it:
  - preflights the user's output_dir (fails BEFORE spending if the path/share is bad),
  - POSTs the task,
  - writes a per-job sidecar (status="submitted") into that output_dir — the
    portable recovery receipt,
  - registers the job in the central registry,
  - ensures the background watcher is running (it polls, downloads to output_dir,
    updates the sidecar, and toasts on completion — no manual polling).

Recover/inspect later with Zerogen_SeedanceFetchTask (by task_id) if ever needed.

Note: the watcher re-resolves the API key from env/.env, so leave `api_key` empty
unless your key is ONLY available as a node override (then the watcher can't see it
— use the blocking gen node instead).
"""

from __future__ import annotations

import json
import time

import aiohttp

from comfy_api.latest import IO

from . import seedance_job_registry as registry
from .api_keys import resolve_api_key
from .nv_byteplus_seedance_gen import (
    _API_BASE,
    _CREATE_PATH,
    _DURATION_MODES,
    _MAX_REF_IMAGES,
    _MAX_REF_VIDEOS,
    _MODELS,
    _RATIOS,
    _RESOLUTIONS,
    _IMAGE_MODES,
    _assert_keyframe_prompt_clean,
    _auto_inject_tags,
    _build_content,
    _measure_video_duration,
    _parse_ref_urls,
    _post_task,
    _resolve_duration,
    _resolve_image_roles,
)
from .seedance_watcher import ensure_watcher_running


class Zerogen_ByteplusSeedanceSubmit(IO.ComfyNode):
    """Fire a Seedance gen and return immediately (non-blocking). A background
    watcher saves the result to your output_dir and toasts when done."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Zerogen_ByteplusSeedanceSubmit",
            display_name="BytePlus Seedance Submit (non-blocking)",
            category="zerogen",
            description=(
                "Submit a Seedance 2.0 gen WITHOUT blocking — returns a task_id in ~2s and frees "
                "ComfyUI. A background watcher polls, saves the video (+ optional last_frame + sidecar) "
                "to output_dir, and toasts on completion. Supports reference + keyframe (first/last "
                "frame) modes. Recover by task_id with Zerogen_SeedanceFetchTask."
            ),
            inputs=[
                IO.String.Input("prompt", multiline=True, default="",
                                tooltip="Generation prompt. @Image1..N / @Video1 tags auto-injected if absent."),
                IO.Combo.Input("mode", options=_IMAGE_MODES, default="multimodal",
                               tooltip="How the image list maps to roles. multimodal: images = reference_image "
                                       "(@Image1..N) + videos = reference_video (covers reference/edit/extend/combined "
                                       "via the prompt verb). first_frame: line 1 = start frame (no ref videos). "
                                       "first_last_frame: line 1 = start, line 2 = end (no ref videos)."),
                IO.String.Input("ref_image_asset_urls", multiline=True, default="",
                                tooltip="Image refs, one asset:// / HTTPS / data: per line. multimodal: 1-9 @Image1..N. "
                                        "first_frame: line 1 = first frame. first_last_frame: line 1 = first, line 2 = last.",
                                optional=True),
                IO.String.Input("ref_video_asset_urls", multiline=True, default="",
                                tooltip="Reference videos (1-3), one per line, @Video1..N order. Ignored in keyframe modes.",
                                optional=True),
                IO.Combo.Input("model", options=list(_MODELS.keys()), default="Seedance 2.0 Pro",
                               tooltip="Seedance model."),
                IO.Combo.Input("resolution", options=_RESOLUTIONS, default="720p", tooltip="480p / 720p / 1080p."),
                IO.Combo.Input("ratio", options=_RATIOS, default="adaptive",
                               tooltip="adaptive recommended for keyframe mode (output AR follows the frames)."),
                IO.Int.Input("duration", default=5, min=-1, max=15, step=1,
                             tooltip="Output seconds (4-15, or -1 for model-auto)."),
                IO.Boolean.Input("generate_audio", default=True, tooltip="Produce synchronized audio."),
                IO.Boolean.Input("watermark", default=False, tooltip="Add watermark."),
                IO.Int.Input("seed", default=-1, min=-1, max=2147483647, control_after_generate=True,
                             tooltip="-1 = random. Folded into the re-run cache key: a fixed seed + identical "
                                     "inputs won't re-submit (double-spend guard); change seed (or any input) to re-fire."),
                IO.Boolean.Input("return_last_frame", default=True,
                                 tooltip="Request the output's last frame PNG (saved if save_last_frame)."),
                # ---- output location ----
                IO.String.Input("output_dir", default="",
                                tooltip="Folder to save results into (absolute, e.g. Z:/.../output). "
                                        "Validated for writability at submit time."),
                IO.String.Input("filename_template", default="{label}_{task_id}",
                                tooltip="Filename stem. Tokens: {label} {task_id} {mode} {date} {time} {datetime}.",
                                optional=True),
                IO.String.Input("label", default="",
                                tooltip="Short human label (e.g. 'jon_bridge') for logs/toasts/filenames.",
                                optional=True),
                IO.Boolean.Input("save_last_frame", default=True,
                                 tooltip="Also save the last_frame PNG beside the mp4.", optional=True),
                IO.Boolean.Input("save_sidecar", default=True,
                                 tooltip="Write a .json receipt (prompt + params + task_id) beside the mp4.",
                                 optional=True),
                # ---- duration-from-ref (optional) ----
                IO.Combo.Input("duration_mode", options=_DURATION_MODES, default="manual",
                               tooltip="manual / model_auto / auto_from_ref (match a ref clip's length).",
                               optional=True),
                IO.Video.Input("ref_video",
                               tooltip="Optional LOCAL clip — measured for duration_mode=auto_from_ref.",
                               optional=True),
                IO.Float.Input("ref_duration_s", default=0.0, min=0.0, max=600.0, step=0.01,
                               tooltip="Manual ref duration (s) fallback for auto_from_ref.", optional=True),
                # ---- watcher ----
                IO.Float.Input("watcher_poll_interval_s", default=10.0, min=2.0, max=60.0,
                               tooltip="How often the background watcher polls (shared across all jobs).",
                               optional=True),
                IO.Combo.Input("watcher_verbosity", options=["quiet", "normal", "debug"], default="normal",
                               tooltip="Console heartbeat detail: quiet (done/fail only) / normal / debug.",
                               optional=True),
                IO.String.Input("api_key", default="",
                                tooltip="Optional ARK_API_KEY override. LEAVE EMPTY for the watcher — it "
                                        "re-resolves from env/.env (an override here is invisible to it).",
                                optional=True),
            ],
            outputs=[
                IO.String.Output(display_name="task_id"),
                IO.String.Output(display_name="sidecar_path"),
                IO.String.Output(display_name="status_json"),
            ],
            is_api_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, seed: int = -1, prompt: str = "", mode: str = "multimodal",
                           ref_image_asset_urls: str = "", ref_video_asset_urls: str = "",
                           model: str = "Seedance 2.0 Pro", resolution: str = "720p",
                           ratio: str = "adaptive", duration: int = 5, generate_audio: bool = True,
                           watermark: bool = False, return_last_frame: bool = True,
                           duration_mode: str = "manual", ref_video=None, ref_duration_s: float = 0.0,
                           **kwargs):  # noqa: ANN001, ANN206
        """V3 IS_CHANGED — double-spend guard. Fingerprints EVERY payload-affecting
        input, so changing the gen (mode/resolution/duration/model/audio/ratio/...) re-submits.
        Output-only fields (output_dir/filename_template/label) and watcher settings are
        DELIBERATELY EXCLUDED — renaming or relocating the output must never spend money
        again. CRITICAL: with duration_mode=auto_from_ref the POST duration comes from the
        MEASURED local ref_video, so we fingerprint the resolved api_duration (review finding:
        swapping the local clip changed the payload but not the old key -> wrong cache hit).
        A fixed seed + identical payload returns the same key -> ComfyUI skips the paid
        re-submit; control_after_generate bumps the seed each queue to fire a fresh job."""
        effective_ref_dur = _measure_video_duration(ref_video) or float(ref_duration_s or 0.0)
        api_duration, _note = _resolve_duration(duration_mode, duration, effective_ref_dur)
        return json.dumps({
            "v": 4, "seed": seed, "prompt": prompt or "", "mode": mode,
            "ref_image_asset_urls": ref_image_asset_urls or "",
            "ref_video_asset_urls": ref_video_asset_urls or "",
            "model": model, "resolution": resolution, "ratio": ratio,
            "duration": duration, "duration_mode": duration_mode,
            "ref_duration_s": round(float(ref_duration_s or 0.0), 3),
            "api_duration": api_duration,   # the value the POST actually uses
            "generate_audio": bool(generate_audio), "watermark": bool(watermark),
            "return_last_frame": bool(return_last_frame),
        }, sort_keys=True, ensure_ascii=False)

    @classmethod
    async def execute(
        cls,
        prompt: str,
        mode: str = "multimodal",
        ref_image_asset_urls: str = "",
        ref_video_asset_urls: str = "",
        model: str = "Seedance 2.0 Pro",
        resolution: str = "720p",
        ratio: str = "adaptive",
        duration: int = 5,
        generate_audio: bool = True,
        watermark: bool = False,
        seed: int = -1,
        return_last_frame: bool = True,
        output_dir: str = "",
        filename_template: str = "{label}_{task_id}",
        label: str = "",
        save_last_frame: bool = True,
        save_sidecar: bool = True,
        duration_mode: str = "manual",
        ref_video=None,
        ref_duration_s: float = 0.0,
        watcher_poll_interval_s: float = 10.0,
        watcher_verbosity: str = "normal",
        api_key: str = "",
        **kwargs,  # sink removed/stale inputs from old saved graphs (don't crash the queue)
    ) -> IO.NodeOutput:
        # ---- resolve + validate (all pre-spend) ----
        all_image_urls = _parse_ref_urls(ref_image_asset_urls, "")
        video_urls = _parse_ref_urls(ref_video_asset_urls, "", kind="video")
        n_videos = len(video_urls)
        # Map the image list to roles per the calling mode (raises pre-spend on bad combos).
        image_urls, first_frame_url, last_frame_url = _resolve_image_roles(mode, all_image_urls, n_videos)
        n_images = len(image_urls)
        keyframe_mode = mode
        if n_images > _MAX_REF_IMAGES:
            raise ValueError(f"Seedance accepts at most {_MAX_REF_IMAGES} reference images; got {n_images}.")
        if n_videos > _MAX_REF_VIDEOS:
            raise ValueError(f"Seedance accepts at most {_MAX_REF_VIDEOS} reference videos; got {n_videos}.")

        final_prompt = (prompt or "").strip()
        _assert_keyframe_prompt_clean(mode, final_prompt)
        if not final_prompt and not (image_urls or video_urls or first_frame_url or last_frame_url):
            raise ValueError("No prompt and no refs — can't submit an empty task.")
        final_prompt = _auto_inject_tags(final_prompt, n_images, n_videos)

        model_id = _MODELS[model]
        effective_ref_dur = _measure_video_duration(ref_video) or float(ref_duration_s or 0.0)
        api_duration, duration_note = _resolve_duration(duration_mode, duration, effective_ref_dur)

        # Preflight output dir BEFORE spending — bad path/share fails here.
        out_dir = registry.preflight_output_dir(output_dir)

        override_key = (api_key or "").strip()
        if override_key:
            # The watcher re-resolves the key from env/.env and CANNOT see a node
            # override — spending with a key the watcher can't follow up on = a
            # stranded paid job. Hard-fail BEFORE the POST unless env matches (review
            # finding: a printed warning is too weak for a money-spending path).
            try:
                env_key = resolve_api_key("", provider="volcengine")
            except Exception:
                env_key = ""
            if (env_key or "").strip() != override_key:
                raise ValueError(
                    "Non-blocking Submit cannot use an api_key override the background watcher "
                    "can't see — it would spend money but the watcher couldn't poll/download the "
                    "result. Put ARK_API_KEY in env/.env (matching this key), leave api_key empty, "
                    "or use the blocking BytePlus Seedance Gen node for an override."
                )
        resolved_key = resolve_api_key(api_key, provider="volcengine")

        # ---- build payload + POST (no poll) ----
        content = _build_content(final_prompt, image_urls, video_urls, first_frame_url or None, last_frame_url or None)
        payload: dict = {
            "model": model_id, "content": content, "resolution": resolution, "ratio": ratio,
            "duration": api_duration, "generate_audio": generate_audio, "watermark": watermark,
            "return_last_frame": return_last_frame,
        }
        if seed != -1:
            payload["seed"] = seed

        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=120)
        connector = aiohttp.TCPConnector(force_close=True, limit=4)
        session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        try:
            create_resp = await _post_task(session, resolved_key, payload)
        finally:
            await session.close()
            await __import__("asyncio").sleep(0.05)
        task_id = create_resp.get("id")
        if not task_id:
            raise RuntimeError(f"Submit returned no task id. Raw: {create_resp}")
        # Log task_id IMMEDIATELY — before path templating / registry / sidecar — so a
        # crash in the post-spend window still leaves the id in the console for manual
        # recovery via FetchTask (review finding: paid-POST-then-crash window).
        print(f"[Zerogen_ByteplusSeedanceSubmit] POST ok — task_id={task_id} (recoverable via FetchTask)")

        # ---- resolve output paths (need task_id) + write receipt + register ----
        lbl = (label or "").strip() or keyframe_mode
        paths = registry.resolve_output_paths(
            output_dir=out_dir, filename_template=filename_template or "{label}_{task_id}",
            label=lbl, task_id=task_id, mode=keyframe_mode,
        )
        now_epoch = time.time()
        from datetime import datetime
        submitted_iso = datetime.now().isoformat(timespec="seconds")

        sidecar_payload = {
            "task_id": task_id,
            "label": lbl,
            "status": "submitted",
            "submitted_at": submitted_iso,
            "request": {
                "endpoint": _API_BASE, "model": model_id, "mode": keyframe_mode,
                "resolution": resolution, "ratio": ratio, "duration_used": api_duration,
                "duration_mode": duration_mode, "duration_source": duration_note,
                "generate_audio": generate_audio, "watermark": watermark, "seed": seed,
                "prompt": final_prompt, "n_reference_images": n_images, "n_reference_videos": n_videos,
                "first_frame": first_frame_url or None, "last_frame": last_frame_url or None,
            },
            "paths": paths,
        }
        if save_sidecar:
            try:
                registry.write_sidecar(paths["sidecar"], sidecar_payload)
            except Exception as e:  # noqa: BLE001 — registry is the authoritative copy
                print(f"[Zerogen_ByteplusSeedanceSubmit] WARNING: sidecar write failed: {e}")

        registry.add_job({
            "task_id": task_id, "label": lbl, "status": "submitted", "mode": keyframe_mode,
            "submitted_at": submitted_iso, "submitted_at_epoch": now_epoch,
            "paths": paths, "save_last_frame": bool(save_last_frame),
        })

        started = ensure_watcher_running(watcher_poll_interval_s, watcher_verbosity)
        print(f"[Zerogen_ByteplusSeedanceSubmit] submitted '{lbl}' -> {task_id} "
              f"(mode={keyframe_mode}, {resolution}, {api_duration}s) -> {paths['video']}"
              f"{'' if started else ' [WARN: watcher not started — server loop unavailable]'}")

        status_obj = {
            "task_id": task_id, "label": lbl, "mode": keyframe_mode, "status": "submitted",
            "watcher_running": started, "output_video": paths["video"],
            "sidecar": paths["sidecar"] if save_sidecar else None,
        }
        return IO.NodeOutput(
            task_id,
            paths["sidecar"] if save_sidecar else "",
            json.dumps(status_obj, indent=2, ensure_ascii=False),
        )


NODE_CLASS_MAPPINGS = {"Zerogen_ByteplusSeedanceSubmit": Zerogen_ByteplusSeedanceSubmit}
NODE_DISPLAY_NAME_MAPPINGS = {"Zerogen_ByteplusSeedanceSubmit": "BytePlus Seedance Submit (non-blocking)"}
