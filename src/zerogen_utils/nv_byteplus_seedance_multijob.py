"""NV BytePlus Seedance Multi-Job — parallel multi-subject single-shot fanout.

The native (BytePlus international) sibling of NV_SeedanceMoyuMultiJob, but
SINGLE-SHOT per job (no chunking): each job is one Seedance gen task (≤15s),
so this runs N subjects (head/body/…) in parallel from one Run. For outputs
longer than Seedance's 15s single-call limit you'd want a chunked variant —
deliberately out of scope here (operator decision 2026-06-12: their ref-gen
clips are short, so chunking is dormant).

Two nodes:
  - NV_ByteplusSeedanceJobConfig — bundle one subject's prompt + asset:// refs
    (+ optional per-job overrides incl. smart timing / retime) into a
    BYTEPLUS_SEEDANCE_JOB_CONFIG bus dict.
  - NV_ByteplusSeedanceMultiJob — take up to 4 job configs, run them concurrently
    (bounded by max_concurrent_jobs), each via the SHARED `_generate_one` core
    that the single gen node uses (zero logic duplication). Soft-fail per job;
    per-slot frames out + aggregated status/task_ids JSON.

Reuses the gen node's reviewed core + resolvers — this module only adds the
fanout + per-job param merge. Separate from the mainland/Moyu nodes.
"""

from __future__ import annotations

import asyncio
import json

from comfy_api.latest import IO
from comfy_api.latest._io import Custom as _IOCustom

from .api_keys import resolve_api_key
from .nv_byteplus_seedance_gen import (
    _DURATION_MODES,
    _MAX_REF_IMAGES,
    _MODELS,
    _RATIOS,
    _RESOLUTIONS,
    _auto_inject_tags,
    _generate_one,
    _measure_video_duration,
    _parse_ref_urls,
    _resolve_duration,
)

try:
    from comfy.model_management import throw_exception_if_processing_interrupted as _check_interrupt  # noqa: F401
except ImportError:
    def _check_interrupt() -> None:
        pass

# Custom bus type — one per-subject job config dict from the builder node.
BYTEPLUS_SEEDANCE_JOB_CONFIG = _IOCustom("BYTEPLUS_SEEDANCE_JOB_CONFIG")

_MAX_JOB_SLOTS = 4   # fixed static slots (ComfyUI dislikes dynamic output counts)


def _is_interrupt(exc: BaseException) -> bool:
    """True if `exc` is a ComfyUI cancel / asyncio cancellation we must re-raise."""
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        return True
    return exc.__class__.__name__ in {
        "InterruptProcessingException",
        "ExecutionInterruptedException",
        "CancelledError",
        "KeyboardInterrupt",
    }


# ---------------------------------------------------------------------------
# Builder node — one per subject
# ---------------------------------------------------------------------------


class NV_ByteplusSeedanceJobConfig(IO.ComfyNode):
    """Bundle one subject's inputs into a BYTEPLUS_SEEDANCE_JOB_CONFIG.

    Wire one per subject (head, body, …): its prompt + pre-registered asset://
    refs. Per-job resolution/ratio/duration/retime overrides are opt-in via
    `override_params`; otherwise the slot inherits the fanout node's shared
    defaults. `ref_duration_s` is always per-job (each ref clip has its own
    length) and drives duration_mode=auto_from_ref.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_ByteplusSeedanceJobConfig",
            display_name="NV BytePlus Seedance Job Config",
            category="NV_Utils/api",
            description=(
                "Bundle one subject's prompt + asset:// refs (+ optional per-job overrides) into a "
                "BYTEPLUS_SEEDANCE_JOB_CONFIG for NV_ByteplusSeedanceMultiJob."
            ),
            inputs=[
                IO.String.Input("prompt", multiline=True, default="",
                                tooltip="Final Seedance prompt for THIS subject (e.g. from NV_PromptRefiner)."),
                IO.String.Input("ref_image_asset_urls", multiline=True, default="",
                                tooltip="Newline-separated image asset:// URLs (1-9, @Image order). Wire NV_ByteplusImageBatchRegister.joined_urls.",
                                optional=True),
                IO.String.Input("ref_image_asset_url", default="",
                                tooltip="Single image ref asset:// URL. Mutually exclusive with the multi-line list.",
                                optional=True),
                IO.String.Input("ref_video_asset_url", default="",
                                tooltip="Optional reference VIDEO asset:// URL (role reference_video).",
                                optional=True),
                IO.Video.Input("ref_video",
                               tooltip="Optional LOCAL source video — measured here for duration_mode=auto_from_ref (works with pre-registered asset:// refs). Wire the same clip you registered. Takes priority over ref_duration_s.",
                               optional=True),
                IO.Float.Input("ref_duration_s", default=0.0, min=0.0, max=600.0, step=0.01,
                               tooltip="Manual fallback duration (s) for auto_from_ref when no ref_video is wired. Drives duration_mode=auto_from_ref.",
                               optional=True),
                IO.String.Input("slot_label", default="",
                                tooltip="Short label (e.g. 'head', 'body') for the status JSON. Defaults to slotNN.",
                                optional=True),
                IO.Boolean.Input("override_params", default=False,
                                 tooltip="When True, use THIS slot's resolution/ratio/duration_mode/duration/slowdown_factor instead of the fanout node's shared defaults."),
                IO.Combo.Input("resolution", options=_RESOLUTIONS, default="720p",
                               tooltip="Per-slot resolution override (only when override_params=True).", optional=True),
                IO.Combo.Input("ratio", options=_RATIOS, default="adaptive",
                               tooltip="Per-slot ratio override (only when override_params=True).", optional=True),
                IO.Combo.Input("duration_mode", options=_DURATION_MODES, default="manual",
                               tooltip="Per-slot duration mode override (only when override_params=True).", optional=True),
                IO.Int.Input("duration", default=5, min=-1, max=15,
                             tooltip="Per-slot manual duration override (only when override_params=True).", optional=True),
                IO.Int.Input("slowdown_factor", default=1, min=1, max=8,
                             tooltip="Per-slot retime restore override (only when override_params=True).", optional=True),
            ],
            outputs=[BYTEPLUS_SEEDANCE_JOB_CONFIG.Output(display_name="job_config")],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        ref_image_asset_urls: str = "",
        ref_image_asset_url: str = "",
        ref_video_asset_url: str = "",
        ref_video=None,
        ref_duration_s: float = 0.0,
        slot_label: str = "",
        override_params: bool = False,
        resolution: str = "720p",
        ratio: str = "adaptive",
        duration_mode: str = "manual",
        duration: int = 5,
        slowdown_factor: int = 1,
    ) -> IO.NodeOutput:
        # Measure the local source clip at build time (works with pre-registered
        # asset:// refs); store the resolved duration as a float in the bus.
        effective_ref_dur = _measure_video_duration(ref_video) or float(ref_duration_s or 0.0)
        cfg = {
            "slot_label": (slot_label or "").strip(),
            "prompt": prompt or "",
            "ref_image_asset_urls": ref_image_asset_urls or "",
            "ref_image_asset_url": ref_image_asset_url or "",
            "ref_video_asset_url": ref_video_asset_url or "",
            "ref_duration_s": float(effective_ref_dur),
            # Override-able fields: None = inherit the fanout node's shared default.
            "resolution": resolution if override_params else None,
            "ratio": ratio if override_params else None,
            "duration_mode": duration_mode if override_params else None,
            "duration": duration if override_params else None,
            "slowdown_factor": slowdown_factor if override_params else None,
        }
        return IO.NodeOutput(cfg)


# ---------------------------------------------------------------------------
# Fanout node
# ---------------------------------------------------------------------------


class NV_ByteplusSeedanceMultiJob(IO.ComfyNode):
    """Run up to 4 single-shot BytePlus Seedance gens concurrently from one Run.

    Each job = one gen task (≤15s) via the shared `_generate_one` core. Bounded
    by max_concurrent_jobs. Soft-fail per job (one subject's failure never kills
    siblings). Per-slot frames out + aggregated status/task_ids JSON.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        inputs = [
            BYTEPLUS_SEEDANCE_JOB_CONFIG.Input(f"job_{i}", optional=True,
                                               tooltip=f"Subject job #{i} from NV_ByteplusSeedanceJobConfig. Head→job_1, body→job_2, …")
            for i in range(1, _MAX_JOB_SLOTS + 1)
        ]
        inputs += [
            IO.Combo.Input("model", options=list(_MODELS.keys()), default="Seedance 2.0 Pro",
                           tooltip="Seedance model (shared)."),
            IO.Combo.Input("resolution", options=_RESOLUTIONS, default="720p",
                           tooltip="Shared default resolution (a job can override)."),
            IO.Combo.Input("ratio", options=_RATIOS, default="adaptive",
                           tooltip="Shared default ratio (a job can override)."),
            IO.Combo.Input("duration_mode", options=_DURATION_MODES, default="manual",
                           tooltip="Shared default duration mode (a job can override). auto_from_ref uses each job's ref_duration_s."),
            IO.Int.Input("duration", default=5, min=-1, max=15,
                         tooltip="Shared default manual duration (a job can override)."),
            IO.Int.Input("slowdown_factor", default=1, min=1, max=8,
                         tooltip="Shared default retime restore factor (a job can override)."),
            IO.Boolean.Input("generate_audio", default=False, tooltip="Request audio (shared)."),
            IO.Boolean.Input("watermark", default=False, tooltip="Watermark output (shared)."),
            IO.Boolean.Input("return_last_frame", default=True, tooltip="Request each output's last frame (shared)."),
            IO.Int.Input("seed", default=-1, min=-1, max=2_147_483_647,
                         tooltip="-1 = random per job. A fixed seed is reused across jobs (parity)."),
            IO.Int.Input("max_concurrent_jobs", default=2, min=1, max=_MAX_JOB_SLOTS,
                         tooltip="How many subject jobs run at once. Each is a separate paid gen — this is the real-money concurrency cap."),
            IO.Combo.Input("failure_policy", options=["error_on_all_failed", "error_on_any_failed", "never_raise"],
                           default="error_on_all_failed",
                           tooltip="error_on_all_failed (default): raise only if every job fails. error_on_any_failed: raise if any fails. never_raise: status_json is authoritative."),
            IO.Boolean.Input("error_on_noop", default=True,
                             tooltip="Raise if zero job slots are wired (catches a misconfigured graph)."),
            IO.Float.Input("poll_interval_s", default=9.0, min=2.0, max=60.0, tooltip="Per-job poll interval (shared)."),
            IO.Float.Input("poll_timeout_s", default=1500.0, min=60.0, max=7200.0, tooltip="Per-job poll timeout (shared)."),
            IO.String.Input("api_key", default="", optional=True,
                            tooltip="Bearer ARK_API_KEY override. Empty → env ARK_API_KEY / VOLCENGINE_ARK_API_KEY / .env."),
        ]
        outputs = [IO.Image.Output(display_name=f"job_{i}_frames") for i in range(1, _MAX_JOB_SLOTS + 1)]
        outputs += [
            IO.String.Output(display_name="status_json"),
            IO.String.Output(display_name="task_ids_json"),
        ]
        return IO.Schema(
            node_id="NV_ByteplusSeedanceMultiJob",
            display_name="NV BytePlus Seedance Multi-Job",
            category="NV_Utils/api",
            description=(
                "Run up to 4 single-shot BytePlus Seedance gens (≤15s each) in parallel from one Run. "
                "Bounded concurrency, soft-fail per job, per-slot frames + aggregated status/task_ids JSON. "
                "Reuses the validated single-gen core."
            ),
            inputs=inputs,
            outputs=outputs,
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model: str = "Seedance 2.0 Pro",
        resolution: str = "720p",
        ratio: str = "adaptive",
        duration_mode: str = "manual",
        duration: int = 5,
        slowdown_factor: int = 1,
        generate_audio: bool = False,
        watermark: bool = False,
        return_last_frame: bool = True,
        seed: int = -1,
        max_concurrent_jobs: int = 2,
        failure_policy: str = "error_on_all_failed",
        error_on_noop: bool = True,
        poll_interval_s: float = 9.0,
        poll_timeout_s: float = 1500.0,
        api_key: str = "",
        job_1=None,
        job_2=None,
        job_3=None,
        job_4=None,
    ) -> IO.NodeOutput:
        active = [(i, j) for i, j in enumerate([job_1, job_2, job_3, job_4]) if isinstance(j, dict)]
        if not active:
            msg = "[NV_ByteplusSeedanceMultiJob] No job slots wired — nothing to run."
            if error_on_noop:
                raise ValueError(msg + " (set error_on_noop=False to no-op silently.)")
            print(msg)
            return cls._empty_output("no active jobs")

        resolved_key = resolve_api_key(api_key, provider="volcengine")
        model_id = _MODELS[model]

        # ---- Phase 1: per-slot resolve + deterministic preflight (no network) ----
        prepared = []  # (slot_idx, label, gen_kwargs)
        for slot_idx, job in active:
            label = (job.get("slot_label") or "").strip() or f"slot{slot_idx + 1:02d}"
            slot_id = f"job_{slot_idx + 1} ({label})"

            image_urls = _parse_ref_urls(job.get("ref_image_asset_urls", ""), job.get("ref_image_asset_url", ""))
            video_url = (job.get("ref_video_asset_url") or "").strip()
            has_video = bool(video_url)
            n_images = len(image_urls)
            if n_images > _MAX_REF_IMAGES:
                raise ValueError(f"[NV_ByteplusSeedanceMultiJob] {slot_id}: at most {_MAX_REF_IMAGES} ref images; got {n_images}.")

            final_prompt = (job.get("prompt") or "").strip()
            if not final_prompt and n_images == 0 and not has_video:
                raise ValueError(f"[NV_ByteplusSeedanceMultiJob] {slot_id}: empty prompt and no refs.")
            final_prompt = _auto_inject_tags(final_prompt, n_images, has_video)

            # per-job override or shared default
            j_res = job.get("resolution") or resolution
            j_ratio = job.get("ratio") or ratio
            j_dmode = job.get("duration_mode") or duration_mode
            j_dur = job.get("duration") if job.get("duration") is not None else duration
            j_slow = job.get("slowdown_factor") or slowdown_factor
            # _resolve_duration validates all modes (raises here in Phase 1 = pre-spend).
            try:
                api_duration, dnote = _resolve_duration(j_dmode, j_dur, float(job.get("ref_duration_s") or 0.0))
            except ValueError as ve:
                raise ValueError(f"[NV_ByteplusSeedanceMultiJob] {slot_id}: {ve}")

            print(f"[NV_ByteplusSeedanceMultiJob] {slot_id}: res={j_res} ratio={j_ratio} dur={api_duration}s "
                  f"[{dnote}] slowdown={j_slow} refs(img={n_images} vid={'y' if has_video else 'n'}).")

            prepared.append((slot_idx, label, dict(
                cls=cls, final_prompt=final_prompt, image_urls=image_urls, video_url=video_url,
                model_id=model_id, resolution=j_res, ratio=j_ratio, api_duration=api_duration,
                generate_audio=generate_audio, watermark=watermark, seed=seed,
                return_last_frame=return_last_frame, slowdown_factor=j_slow,
                poll_interval_s=poll_interval_s, poll_timeout_s=poll_timeout_s, resolved_key=resolved_key,
                log_tag=f"MultiJob:{slot_id}",
            )))

        # ---- Phase 2: bounded-concurrency dispatch, soft-fail per job ----
        sem = asyncio.Semaphore(int(max_concurrent_jobs))

        async def _run_slot(slot_idx, label, gen_kwargs):
            async with sem:
                try:
                    r = await _generate_one(**gen_kwargs)
                    return {"slot": slot_idx + 1, "label": label, "status": "success",
                            "task_id": r["task_id"], "frames": r["frames"],
                            "output_frames": r["frame_count"], "raw_frames": r["raw_frame_count"],
                            "retimed": r["retimed"], "fps": r["fps"], "error": None}
                except BaseException as exc:  # noqa: BLE001 — interrupts re-raised
                    if _is_interrupt(exc):
                        raise
                    if not isinstance(exc, Exception):
                        raise
                    print(f"[NV_ByteplusSeedanceMultiJob] job slot {slot_idx + 1} ({label}) FAILED — {exc.__class__.__name__}: {exc}")
                    return {"slot": slot_idx + 1, "label": label, "status": "failed",
                            "task_id": None, "frames": None, "output_frames": 0, "raw_frames": 0,
                            "retimed": False, "fps": 0.0, "error": f"{exc.__class__.__name__}: {exc}"}

        outcomes = await asyncio.gather(*[_run_slot(s, lb, gk) for (s, lb, gk) in prepared])

        # ---- Phase 3: assemble (slot-indexed) ----
        by_slot = {o["slot"]: o for o in outcomes}
        frame_outputs = [by_slot[i]["frames"] if i in by_slot else None for i in range(1, _MAX_JOB_SLOTS + 1)]

        succeeded = [o for o in outcomes if o["status"] == "success"]
        failed = [o for o in outcomes if o["status"] == "failed"]
        status_obj = {
            "active_jobs": len(outcomes),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "max_concurrent_jobs": int(max_concurrent_jobs),
            "jobs": [{k: v for k, v in o.items() if k != "frames"} for o in outcomes],
        }
        task_ids_obj = {str(o["slot"]): {"label": o["label"], "task_id": o["task_id"]} for o in outcomes}

        print(f"[NV_ByteplusSeedanceMultiJob] DONE — {len(succeeded)}/{len(outcomes)} job(s) succeeded, {len(failed)} failed.")

        if failed:
            fail_summary = "; ".join(f"slot {o['slot']} ({o['label']}): {o['error']}" for o in failed)
            if failure_policy == "error_on_any_failed" or (failure_policy == "error_on_all_failed" and not succeeded):
                raise RuntimeError(
                    f"[NV_ByteplusSeedanceMultiJob] {len(failed)}/{len(outcomes)} job(s) failed "
                    f"({failure_policy}): {fail_summary}. See status_json + task_ids_json "
                    f"(succeeded jobs' frames are on their outputs; recover failed via NV_SeedanceFetchTask)."
                )

        return IO.NodeOutput(
            *frame_outputs,
            json.dumps(status_obj, indent=2, ensure_ascii=False),
            json.dumps(task_ids_obj, indent=2, ensure_ascii=False),
        )

    @classmethod
    def _empty_output(cls, reason: str) -> IO.NodeOutput:
        return IO.NodeOutput(*([None] * _MAX_JOB_SLOTS), json.dumps({"active_jobs": 0, "note": reason}, indent=2), "{}")


NODE_CLASS_MAPPINGS = {
    "NV_ByteplusSeedanceJobConfig": NV_ByteplusSeedanceJobConfig,
    "NV_ByteplusSeedanceMultiJob": NV_ByteplusSeedanceMultiJob,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ByteplusSeedanceJobConfig": "NV BytePlus Seedance Job Config",
    "NV_ByteplusSeedanceMultiJob": "NV BytePlus Seedance Multi-Job",
}
