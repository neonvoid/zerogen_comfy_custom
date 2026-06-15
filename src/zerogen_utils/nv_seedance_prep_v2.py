"""NV Seedance Prep V2 — mode-aware tensor-in preprocessor for Seedance 2.0.

Clean tensor-in interface. Accepts IMAGE / VIDEO inputs via standard ComfyUI
wiring (no URLs, no batching concerns — upstream nodes handle batch shaping).

The 3 mutually-exclusive image modes from the Volcengine API are auto-inferred
from which input slots are wired:

  - Mode A (first_frame):      only `first_frame` wired
  - Mode B (bridge):           `first_frame` AND `last_frame` wired
  - Mode C (multimodal):       only `reference_images` wired (1-9 frames)
  - text_only:                 no image slots wired

Validation fires at both graph-build time (VALIDATE_INPUTS — catches mode
conflicts before the user queues) and inside execute() as a safety net.

Outputs a `SEEDANCE_UPLOAD_CONFIG_V2` carrying role-tagged content. Consumed
by NV_SeedanceNativeRefVideo_V2.

Paper note: per arxiv 2604.14148 Table 27, Seedance 2.0 is architecturally
biased toward dynamic motion at the cost of first-frame fidelity in motion-ref
mode. For strict identity preservation, prefer Mode A (first_frame) — this
node makes that an explicit choice rather than a URL-slot guessing game.
"""

from __future__ import annotations

import json
from fractions import Fraction

import torch
import torch.nn.functional as F
from comfy_api.latest import IO, Input, InputImpl
from comfy_api.latest._util.video_types import VideoComponents

from .nv_seedance_upload_utils import (
    MAX_MULTIMODAL_IMAGES,
    MODE_BRIDGE,
    MODE_FIRST_FRAME,
    MODE_MULTIMODAL,
    MODE_TEXT_ONLY,
    SEEDANCE_UPLOAD_CONFIG_V2,
    build_config,
    infer_mode,
    upload_image_cached,
    upload_video_cached,
    validate_mode,
)


# Volcengine ref-video pixel budget (same as V1 prep)
_REF_VIDEO_PIXEL_MIN = 409_600
_REF_VIDEO_PIXEL_MAX = 2_086_876


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _take_first_frame(image: torch.Tensor, slot_name: str) -> torch.Tensor:
    """Coerce IMAGE to [1,H,W,C], warning if upstream passed a batch."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(
            f"[NV_SeedancePrep_V2] {slot_name} must be an IMAGE [B,H,W,C] or [H,W,C], "
            f"got shape {tuple(image.shape)}"
        )
    if image.shape[0] > 1:
        print(
            f"[NV_SeedancePrep_V2] warning: {slot_name} received batch size {image.shape[0]}, "
            f"using frame [0] only. Wire an Image Batch or Image Select upstream if you "
            f"want a different frame."
        )
        image = image[:1]
    return image


def _even_dims(h: int, w: int) -> tuple[int, int]:
    return h + (h % 2), w + (w % 2)


def _clamp_video_frames_to_budget(frames: torch.Tensor) -> tuple[torch.Tensor, bool, tuple[int, int]]:
    """Bilinear-downscale ref video frames so per-frame pixels fit Volcengine budget."""
    h, w = frames.shape[1], frames.shape[2]
    pixels = h * w
    if _REF_VIDEO_PIXEL_MAX and pixels > _REF_VIDEO_PIXEL_MAX:
        scale = (_REF_VIDEO_PIXEL_MAX / pixels) ** 0.5
        new_h = max(2, int(h * scale))
        new_w = max(2, int(w * scale))
        new_h, new_w = _even_dims(new_h, new_w)
        x = frames.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1), True, (new_h, new_w)
    if _REF_VIDEO_PIXEL_MIN and pixels < _REF_VIDEO_PIXEL_MIN:
        raise ValueError(
            f"Reference video too small: {w}x{h} = {pixels:,}px. "
            f"Minimum {_REF_VIDEO_PIXEL_MIN:,}px (~{int(_REF_VIDEO_PIXEL_MIN ** 0.5)}x"
            f"{int(_REF_VIDEO_PIXEL_MIN ** 0.5)}). Upscale upstream — auto-upscale not supported."
        )
    new_h, new_w = _even_dims(h, w)
    if (new_h, new_w) != (h, w):
        x = frames.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1), True, (new_h, new_w)
    return frames, False, (h, w)


def _build_preview(
    mode: str,
    first_frame: torch.Tensor | None,
    last_frame: torch.Tensor | None,
    reference_images: torch.Tensor | None,
    video_first_frame: torch.Tensor | None,
) -> torch.Tensor:
    """Mode-aware preview assembly."""
    tiles: list[torch.Tensor] = []

    if mode == MODE_FIRST_FRAME and first_frame is not None:
        tiles.append(first_frame)
    elif mode == MODE_BRIDGE and first_frame is not None and last_frame is not None:
        tiles.extend([first_frame, last_frame])
    elif mode == MODE_MULTIMODAL and reference_images is not None:
        for i in range(reference_images.shape[0]):
            tiles.append(reference_images[i:i + 1])

    if video_first_frame is not None:
        tiles.append(video_first_frame)

    if not tiles:
        return torch.zeros(1, 64, 64, 3)

    # Pad all tiles to the max height, then center-pad width to max width.
    target_h = max(t.shape[1] for t in tiles)
    resized: list[torch.Tensor] = []
    for t in tiles:
        h, w = t.shape[1], t.shape[2]
        if h != target_h:
            scale = target_h / h
            new_w = max(1, round(w * scale))
            x = t.permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(target_h, new_w), mode="bilinear", align_corners=False)
            resized.append(x.permute(0, 2, 3, 1))
        else:
            resized.append(t)

    max_w = max(r.shape[2] for r in resized)
    batched: list[torch.Tensor] = []
    for r in resized:
        rw = r.shape[2]
        if rw != max_w:
            pad = torch.zeros(1, target_h, max_w, 3)
            x_off = (max_w - rw) // 2
            pad[:, :, x_off:x_off + rw, :] = r
            batched.append(pad)
        else:
            batched.append(r)
    return torch.cat(batched, dim=0)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class NV_SeedancePrep_V2(IO.ComfyNode):
    """Tensor-in Seedance 2.0 preprocessor. Mode inferred from wired inputs."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_SeedancePrep_V2",
            display_name="NV Seedance Prep V2",
            category="NV_Utils/api",
            description=(
                "Tensor-in Seedance 2.0 preprocessor. Wire inputs to pick a mode:\n"
                "  • first_frame only → i2v mode (best for strict identity lock)\n"
                "  • first_frame + last_frame → bridge mode\n"
                "  • reference_images (1-9) → multimodal mode\n"
                "  • nothing wired → text-only\n"
                "Emits a role-tagged upload_config for NV Seedance Native Ref Video V2."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Draft prompt passed through. Refine via NV Prompt Refiner before the API node.",
                ),
                IO.Image.Input(
                    "first_frame",
                    tooltip=(
                        "Wire a single-frame IMAGE here for Mode A (i2v) or Mode B (bridge). "
                        "Mutually exclusive with reference_images."
                    ),
                    optional=True,
                ),
                IO.Image.Input(
                    "last_frame",
                    tooltip=(
                        "Wire a single-frame IMAGE here with first_frame for Mode B (bridge mode: "
                        "Seedance generates motion from first to last frame). Requires first_frame."
                    ),
                    optional=True,
                ),
                IO.Image.Input(
                    "reference_images",
                    tooltip=(
                        "Wire a batched IMAGE [N,H,W,C] here for Mode C (multimodal ref). "
                        f"1-{MAX_MULTIMODAL_IMAGES} frames. Mutually exclusive with first_frame/last_frame."
                    ),
                    optional=True,
                ),
                IO.Video.Input(
                    "reference_video",
                    tooltip=(
                        "Optional reference_video — combinable with any image mode. "
                        f"Auto-downscaled to fit pixel budget ({_REF_VIDEO_PIXEL_MAX:,}px max per frame)."
                    ),
                    optional=True,
                ),
            ],
            outputs=[
                SEEDANCE_UPLOAD_CONFIG_V2.Output(display_name="upload_config"),
                IO.Image.Output(display_name="preview"),
                IO.String.Output(display_name="mode"),
                IO.String.Output(display_name="prompt"),
                IO.String.Output(display_name="info"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    # Graph-build-time validator. Returns True on OK, error string on fail.
    # ComfyUI highlights the node red with the message before execution.
    @classmethod
    def VALIDATE_INPUTS(
        cls,
        prompt=None,
        first_frame=None,
        last_frame=None,
        reference_images=None,
        reference_video=None,
    ):
        has_first = first_frame is not None
        has_last = last_frame is not None
        if reference_images is None:
            ref_count = 0
        elif hasattr(reference_images, "shape") and reference_images.ndim >= 3:
            ref_count = reference_images.shape[0] if reference_images.ndim == 4 else 1
        else:
            ref_count = 0

        ok, msg = validate_mode(
            has_first_frame=has_first,
            has_last_frame=has_last,
            reference_images_count=ref_count,
            has_reference_video=reference_video is not None,
        )
        if not ok:
            return msg
        return True

    @classmethod
    async def execute(
        cls,
        prompt: str,
        first_frame: Input.Image | None = None,
        last_frame: Input.Image | None = None,
        reference_images: Input.Image | None = None,
        reference_video: Input.Video | None = None,
    ) -> IO.NodeOutput:
        # --- normalize inputs ---
        ff = _take_first_frame(first_frame, "first_frame") if first_frame is not None else None
        lf = _take_first_frame(last_frame, "last_frame") if last_frame is not None else None

        ri = reference_images
        if ri is not None:
            if ri.ndim == 3:
                ri = ri.unsqueeze(0)
            if ri.ndim != 4:
                raise ValueError(
                    f"[NV_SeedancePrep_V2] reference_images must be IMAGE [B,H,W,C] or [H,W,C], "
                    f"got shape {tuple(ri.shape)}"
                )

        has_first = ff is not None
        has_last = lf is not None
        ref_count = ri.shape[0] if ri is not None else 0

        # Re-validate at execute (VALIDATE_INPUTS is best-effort; catches most but not all)
        ok, msg = validate_mode(
            has_first_frame=has_first,
            has_last_frame=has_last,
            reference_images_count=ref_count,
            has_reference_video=reference_video is not None,
        )
        if not ok:
            raise ValueError(f"[NV_SeedancePrep_V2] {msg}")

        mode = infer_mode(has_first, has_last, ref_count > 0)
        print(f"[NV_SeedancePrep_V2] Mode: {mode} | first={has_first} last={has_last} "
              f"refs={ref_count} video={'yes' if reference_video is not None else 'no'}")

        # --- collect + validate reference video ---
        uploaded_video_url: str | None = None
        ref_video_obj: Input.Video | None = None
        ref_w = ref_h = 0
        ref_dur = 0.0
        video_first_frame: torch.Tensor | None = None

        if reference_video is not None:
            # VIDEO input is already encoded. Check its dimensions against Volcengine budget.
            try:
                ref_w, ref_h = reference_video.get_dimensions()
                ref_dur = float(reference_video.get_duration())
            except Exception:
                ref_w = ref_h = 0
                ref_dur = 0.0

            pixels = ref_w * ref_h
            if pixels and pixels > _REF_VIDEO_PIXEL_MAX:
                raise ValueError(
                    f"reference_video is {ref_w}x{ref_h} = {pixels:,}px, over "
                    f"{_REF_VIDEO_PIXEL_MAX:,}px Volcengine budget. Re-encode upstream."
                )
            if pixels and pixels < _REF_VIDEO_PIXEL_MIN:
                raise ValueError(
                    f"reference_video is {ref_w}x{ref_h} = {pixels:,}px, under "
                    f"{_REF_VIDEO_PIXEL_MIN:,}px minimum. Use a larger source."
                )
            if ref_dur and (ref_dur < 1.8 or ref_dur > 15.1):
                raise ValueError(
                    f"reference_video duration {ref_dur:.2f}s outside API range [1.8s, 15.1s]."
                )
            ref_video_obj = reference_video

            # Best-effort extract first frame for preview
            try:
                components = reference_video.get_components()
                if components.images.shape[0] > 0:
                    video_first_frame = components.images[0:1]
            except Exception:
                video_first_frame = None

        # --- uploads (all with MD5 dedup) ---
        image_items: list[tuple[str, str]] = []   # (url, role)

        if mode == MODE_FIRST_FRAME:
            url = await upload_image_cached(cls, ff, wait_label="Uploading @Image1 (first_frame)")
            image_items.append((url, "first_frame"))
        elif mode == MODE_BRIDGE:
            url1 = await upload_image_cached(cls, ff, wait_label="Uploading first_frame")
            url2 = await upload_image_cached(cls, lf, wait_label="Uploading last_frame")
            image_items.append((url1, "first_frame"))
            image_items.append((url2, "last_frame"))
        elif mode == MODE_MULTIMODAL:
            for i in range(ref_count):
                frame = ri[i:i + 1]
                url = await upload_image_cached(
                    cls, frame, wait_label=f"Uploading @Image{i + 1} (reference_image)"
                )
                image_items.append((url, "reference_image"))

        if ref_video_obj is not None:
            uploaded_video_url = await upload_video_cached(
                cls, ref_video_obj, wait_label="Uploading @Video1 (reference_video)"
            )

        # --- build config ---
        provenance = {
            "encode_source": "tensor_upload_v2",
            "ref_video_dimensions": [ref_w, ref_h] if uploaded_video_url else None,
            "ref_video_duration_s": round(ref_dur, 3) if uploaded_video_url else None,
        }
        config = build_config(
            mode=mode,
            image_items=image_items,
            video_url=uploaded_video_url,
            prompt=prompt,
            provenance=provenance,
        )

        # --- preview ---
        preview = _build_preview(mode, ff, lf, ri, video_first_frame)

        info = json.dumps(
            {
                "mode": mode,
                "n_images": len(image_items),
                "image_roles": [r for _, r in image_items],
                "has_reference_video": uploaded_video_url is not None,
                "ref_video_dimensions": provenance["ref_video_dimensions"],
                "ref_video_duration_s": provenance["ref_video_duration_s"],
                "prompt_length": len(prompt),
            },
            indent=2,
        )

        print(f"[NV_SeedancePrep_V2] Ready: mode={mode}, {len(image_items)} image(s), "
              f"video={'yes' if uploaded_video_url else 'no'}")

        return IO.NodeOutput(config, preview, mode, prompt, info)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_SeedancePrep_V2": NV_SeedancePrep_V2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SeedancePrep_V2": "NV Seedance Prep V2",
}
