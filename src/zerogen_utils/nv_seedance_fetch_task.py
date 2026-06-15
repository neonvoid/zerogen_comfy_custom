"""NV Seedance Fetch Task — retrieve a finished (or still-running) task by ID.

Companion to NV_SeedanceNativeRefVideo_V2. Use when:
  - Native_V2's poll timed out and you want to retrieve the eventually-finished video
  - You submitted a task earlier and want to download it later from a fresh graph
  - You want to check status of a long-running task without re-submitting

Inputs: task_id (from Native_V2's log or output), poll params, optional api_key.
Outputs: same signature as Native_V2 (video, images, last_frame, fps, frames, task_id, info)
minus final_prompt (not retrievable from task_id alone).

This node does NOT submit anything — no cost to run. Polls the status endpoint
and downloads on success. If the task is still running, polls until done or
timeout (same behavior as Native_V2's poll loop).
"""
from __future__ import annotations

import asyncio
import json
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
from .nv_seedance_native_v2 import _poll_task


class NV_SeedanceFetchTask(IO.ComfyNode):
    """Retrieve a Seedance task by ID. No cost — just fetches existing result."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NV_SeedanceFetchTask",
            display_name="NV Seedance Fetch Task",
            category="NV_Utils/api",
            description=(
                "Fetch a Seedance task by task_id. Use to retrieve videos whose generation "
                "outlived Native_V2's poll_timeout, or to re-download within the 24h signed-URL "
                "window. Submits nothing — no API cost."
            ),
            inputs=[
                IO.String.Input(
                    "task_id",
                    multiline=False,
                    default="",
                    force_input=False,
                    tooltip=(
                        "Seedance task ID like 'cgt-20260425031940-gl59p'. "
                        "From Native_V2's console log or task_id output slot."
                    ),
                ),
                IO.Float.Input(
                    "poll_interval_s",
                    default=15.0,
                    min=2.0,
                    max=120.0,
                    step=1.0,
                    tooltip="Seconds between status polls. Slower is fine here — just recovery.",
                ),
                IO.Float.Input(
                    "poll_timeout_s",
                    default=1800.0,
                    min=10.0,
                    max=7200.0,
                    step=30.0,
                    tooltip=(
                        "Max seconds to wait. Set short (60-120s) if you just want a one-shot "
                        "status check — will return with error if still running. Set long "
                        "(1800-7200s) to block until completion."
                    ),
                ),
                IO.Boolean.Input(
                    "return_last_frame",
                    default=True,
                    tooltip="Fetch the last_frame PNG as a separate IMAGE output.",
                ),
                IO.String.Input(
                    "api_key",
                    default="",
                    tooltip="Optional override. Empty → env VOLCENGINE_ARK_API_KEY / .env.",
                ),
            ],
            outputs=[
                IO.Video.Output(display_name="video"),
                IO.Image.Output(display_name="images"),
                IO.Image.Output(display_name="last_frame"),
                IO.Float.Output(display_name="output_fps"),
                IO.Int.Output(display_name="output_frames"),
                IO.String.Output(display_name="task_id"),
                IO.String.Output(display_name="api_metadata"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        task_id: str,
        poll_interval_s: float,
        poll_timeout_s: float,
        return_last_frame: bool,
        api_key: str,
    ):
        task_id = (task_id or "").strip()
        if not task_id:
            raise ValueError(
                "[NV_SeedanceFetchTask] task_id is empty. Paste the ID from "
                "Native_V2's log (line starts with 'Task submitted: cgt-...')."
            )

        resolved_key = resolve_api_key(api_key, provider="volcengine")
        print(f"[NV_SeedanceFetchTask] Fetching task {task_id} (interval={poll_interval_s}s, "
              f"timeout={poll_timeout_s}s)")

        t_start = time.time()
        timeout_cfg = aiohttp.ClientTimeout(connect=30, sock_read=120)
        connector = aiohttp.TCPConnector(force_close=True)
        session = aiohttp.ClientSession(timeout=timeout_cfg, connector=connector)
        try:
            final_resp = await _poll_task(
                session, resolved_key, task_id, poll_interval_s, poll_timeout_s
            )
        finally:
            await session.close()
            await asyncio.sleep(0.1)

        status = final_resp.get("status")
        if status != "succeeded":
            err = final_resp.get("error") or {}
            raise RuntimeError(
                f"[NV_SeedanceFetchTask] task ended status={status}. "
                f"error.code={err.get('code')!r} error.message={err.get('message')!r}. "
                f"task_id={task_id}"
            )

        resp_content = final_resp.get("content") or {}
        video_url = resp_content.get("video_url")
        last_frame_url = resp_content.get("last_frame_url")
        if not video_url:
            raise RuntimeError(
                f"[NV_SeedanceFetchTask] task succeeded but content.video_url missing. "
                f"Raw: {final_resp}"
            )

        output_video = await download_url_to_video_output(video_url)
        try:
            components = output_video.get_components()
            out_images = components.images
            out_fps = float(components.frame_rate)
            out_frames = int(out_images.shape[0])
        except Exception as e:
            print(f"[NV_SeedanceFetchTask] Warning: frame decode failed: {e}")
            out_images = torch.zeros(1, 64, 64, 3)
            out_fps = 0.0
            out_frames = 0

        last_frame_tensor = torch.zeros(1, 64, 64, 3)
        last_frame_fetched = False
        if return_last_frame and last_frame_url:
            try:
                png_bytes = BytesIO()
                await download_url_to_bytesio(
                    last_frame_url, png_bytes, timeout=30, max_retries=3, cls=cls
                )
                png_bytes.seek(0)
                img = Image.open(png_bytes).convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                last_frame_tensor = torch.from_numpy(arr).unsqueeze(0)
                last_frame_fetched = True
                print(f"[NV_SeedanceFetchTask] last_frame fetched: "
                      f"{tuple(last_frame_tensor.shape)}")
            except Exception as e:
                print(f"[NV_SeedanceFetchTask] Warning: last_frame fetch failed: {e}")

        elapsed = time.time() - t_start
        info = {
            "task_id": task_id,
            "status": status,
            "elapsed_fetch_sec": round(elapsed, 1),
            "output_fps": out_fps,
            "output_frames": out_frames,
            "last_frame_fetched": last_frame_fetched,
            "created_at": final_resp.get("created_at"),
            "updated_at": final_resp.get("updated_at"),
            "usage": final_resp.get("usage"),
            "video_url": video_url,  # keep for manual re-download within 24h
            "last_frame_url": last_frame_url,
        }
        print(f"[NV_SeedanceFetchTask] OK — {out_frames} frames @ {out_fps}fps, "
              f"fetched in {elapsed:.1f}s")

        return IO.NodeOutput(
            output_video, out_images, last_frame_tensor, out_fps, out_frames,
            task_id, json.dumps(info, indent=2, ensure_ascii=False),
        )


NODE_CLASS_MAPPINGS = {
    "NV_SeedanceFetchTask": NV_SeedanceFetchTask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SeedanceFetchTask": "NV Seedance Fetch Task",
}
