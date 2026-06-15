# zerogen_comfy_custom — Intro & Best Practices

A practical guide to the native **BytePlus / Volcengine Ark Seedance 2.0** nodes in
this pack: how the pipeline fits together, how to set it up, and the prompting +
reference-video practices that actually move quality. New to the pack? Read top to
bottom once; after that it's a reference.

> Scope: this pack is the **native ark-direct** path only (asset library + native
> generation). LLM/VLM prompt helpers and the Comfy-proxy / Moyu Seedance paths
> live in `NV_Comfy_Utils`.

---

## 1. The mental model

Seedance 2.0 generation has **two API planes** — keep them straight, they use
different credentials:

| Plane | What | Credential |
|---|---|---|
| **Generation** | Create/poll a video task | **Bearer** `ARK_API_KEY` |
| **Asset library** | Register reference images/videos so the model can use them | **Access Key (AK/SK)** `ARK_ACCESS_KEY` + `ARK_SECRET_KEY` |

The canonical flow is **register → wait Active → generate**:

```
[Zerogen_ByteplusImageBatchRegister]  →  asset://… ids
            │  (poll until status = Active)
            ▼
[Zerogen_ByteplusSeedanceGen]  ──→  video
   text prompt refers to assets as @Image1 / @Video1 …
```

---

## 2. Setup

1. Clone into `ComfyUI/custom_nodes/`, then `pip install -r requirements.txt`.
2. Copy `.env.example` → `.env` and fill it in (see that file for every variable).
   - `ARK_API_KEY` — generation.
   - `ARK_ACCESS_KEY` / `ARK_SECRET_KEY` — asset library (different from the API key!).
   - `B2_*` — optional, only if you stage local media to a URL before registering.
3. Restart ComfyUI. The nodes appear in the node menu under **`zerogen`** (search "Seedance" or "Byteplus").

---

## 3. The asset → generation flow (and its #1 gotcha)

- **CreateAsset is asynchronous.** A registered asset isn't usable until its status
  becomes `Active`. The register nodes poll for you; don't wire an asset into a gen
  before it's Active.
- **`ProjectName` isolation — the most common "it vanished" bug.** The asset library
  is isolated per project. If you register an asset under one `ProjectName` and
  generate with an endpoint in another, the asset "uploads fine but can't be found."
  **Keep registration and generation in the same project** (default is `default`).
- **Reference assets in the prompt by index, never by id.** Use `@Image1`, `@Video1`,
  `@Audio1` — the number is the asset's position among same-type assets in the request.
  Putting a raw `asset://…` id in the prompt text does **not** work.
- Image asset limits: 300–6000 px per side, aspect ratio 0.4–2.5, < 30 MB,
  jpeg/png/webp/bmp/tiff/gif/heic.

---

## 4. SD2 prompting best practices

Seedance reads a prompt as a **spatial layer** (what's in frame) + a **temporal
layer** (how it changes). Write it in that order: *who + doing what → where +
atmosphere → how the camera moves → style/quality*.

**The proven swap/restyle pattern** (lean, reference-anchored — this consistently
beats heavier prompts):

```
@Video1 is reference for camera framing, scene composition, camera/character movement.
Replace the <originals> with the <subjects> from @Image1 and @Image2.
The left <subject>'s identity, <2–4 key features> are entirely defined by @Image1.
The right <subject>'s identity, <2–4 key features> are entirely defined by @Image2.
Render the entire scene and environment in the <style> visual style of @Image1 and @Image2.
```


**Task-verb rule (avoids the #1 misrouting bug):** for an *edit* or *extend* task,
refer to the clip as `@Video1` directly — do **not** write "reference @Video1", which
re-routes it to a different (reference) task.

---

## 5. Reference-video best practices

The reference video is a **control signal, not a picture** — it should carry only
what you want taken from it (motion/camera/composition), and as little appearance as
possible.

This is all so far still WIP and just from reading the documentation and the tests ive done up till 06/15/2026

-  **Greybox / visually-simple source beats a finished photoreal render for
  swap+restyle.** A finished/busy/photoreal source *leaks* its look into the output
  as a thin "filter-on-top" restyle; a neutral source leaves the output open to your
  prompt + image refs. (Community-corroborated; mechanism mirrors VACE/Wan, where raw
  RGB is abstracted to depth/pose/grey to stop appearance leaking.)
- Use an RGB render only when you *want* to keep the source's color/lighting/materials.
- Make geometry **legible** — clear silhouettes / light shading so the model can parse
  subjects; avoid frantic, blurred motion (it degrades the control signal).
- **Match the reference length to the output** to avoid jump-cut artifacts.
- Hard limits: reference **videos 0–3**, **images 0–9**, **audio 0–3**; at least one
  image or video required (audio alone is not allowed). Video-extend: ≤ 3 clips,
  total ≤ 15 s.

---

## 6. Model / limits cheat-sheet

| Model | Resolutions | Duration |
|---|---|---|
| Seedance 2.0 **Pro** (`dreamina-seedance-2-0-260128`) | 480p / 720p / **1080p** | 4–15 s |
| Seedance 2.0 **Fast** (`-fast-260128`) | 480p / 720p **(no 1080p)** | 4–15 s |

- 24 fps, `.mp4`. **Fast does not do 1080p** — selecting it will error.
- Generating *longer than your reference clip* makes the model invent extra time with a
  visible jump — match output duration to the source.
- `ap-southeast-1` is the only region hosting Seedance video.

---

## 7. Example workflows & renders

> _To be added — drop exported `.json` workflows under `examples/` and renders/montages
> here._

- **Example 1 — Register + single generation:** _(workflow + render placeholder)_
- **Example 2 — Two-subject swap (greybox source):** _(workflow + render placeholder)_
- **Example 3 — Multi-job parallel fanout (head/body):** _(workflow + render placeholder)_
