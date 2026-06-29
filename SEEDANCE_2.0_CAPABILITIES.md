# Seedance 2.0 — Capabilities, Tasks, Roles (API-evidence-backed)

**Compiled 2026-06-28 · last updated 2026-06-29** by scraping the official BytePlus ModelArk docs (see footnote).
Every value here is a literal quote/figure from the docs unless explicitly tagged
`[RUNTIME]` (our own test finding, not in the docs). No inference.

---

## Working hypothesis / mental model (read this first)

> **Seedance 2.0 is a single generative endpoint that re-synthesizes the whole frame
> every time. It REPRODUCES what you reference and CHANGES what you name — but it
> never preserves source pixels, and it reliably holds only ONE complex change per
> shot. "Modes" are prompt-verb routing into the SAME generator, not different
> operations — so they converge on easy tasks and fail identically on hard ones.**

Five pillars (evidence-backed):
1. **One endpoint, no mode switch** — `/contents/generations/tasks`, no mode param;
   behavior is driven by `role` + prompt verbs. Our `reference/edit/extend/bridge`
   labels are *client-side framing*, not API modes.
2. **Always regenerates — zero pixel preservation.** No mask input, no lossless/
   in-place; docs state re-processing *degrades* pixels (2298881 L201). "Regional
   repainting/restoration" exists (2298881 L1636) but is *generative*, not pixel-locked.
3. **One dominant change per shot** — stacking two complex/temporal effects (camera
   move + time-lapse) drops one (2222480 L91; runtime).
4. **It reproduces what you reference** — swaps "keep the background" because scene/
   motion/camera come from `@Video1` and are *faithfully re-rendered*, not preserved.
5. **Modes converge on simple tasks** — edit ≈ reference for a single-change swap
   (A/B confirmed); the verb is cosmetic when the task is one clean change.

Falsifiable predictions: one clean change succeeds in any mode; stacking two complex
changes fails in every mode; no prompt makes output pixel-match the input; the
`first_frame/last_frame` role is the only smooth-transition path (untested in our gen node).

Practical rule:
| Goal | Tool |
|---|---|
| ONE generative change, faithful re-render of the rest OK | **Seedance** (any mode — pick by verb) |
| Keep a region **pixel-exact** | **crop + stitch** pipeline (composite onto original) |
| Camera move + keep an existing effect | **post** (2D reframe on rendered footage) |
| Smooth keyframe transition | **first/last-frame role** — needs gen-node support (open) |

---

## Models (Seedance 2.0 series)

| Model | Model ID (literal) | 1080p | 4K |
|---|---|---|---|
| **2.0 Pro** | `dreamina-seedance-2-0-260128` | ✅ | ✅ (Pro only) |
| **2.0 Fast** | `dreamina-seedance-2-0-fast-260128` | ❌ | ❌ |
| **2.0 Mini** | `dreamina-seedance-2-0-mini-260615` | ❌ | ❌ |

"The three models support largely the same features; the difference is the quality/cost
trade-off — Pro = highest quality, Fast = balance, Mini = best cost performance"
(2291680 L37-40). "Only Seedance 2.0 supports 4K" (2298881 L67); "Fast and Mini do not
support 1080p" (L68).

## Endpoints

| Action | Method + path |
|---|---|
| Base URL | `https://ark.ap-southeast.bytepluses.com/api/v3` |
| Create task | `POST /contents/generations/tasks` |
| Query task | `GET  /contents/generations/tasks/{id}` |

Region: `ap-southeast` only. Task data (status, video URL) retained **24h** then cleared
(2298881 L171).

## Tasks — capability + official prompt pattern

| Task | What it does (quoted) | Official prompt pattern | Note |
|---|---|---|---|
| **Multimodal reference** | "Extract elements (subject, style, scene, sound effects) to generate a **brand-new video**" | Image: `Reference <Subject_N> in <Image_N> to generate…` · Video: `Reference <Action/Camera_movement/Style/Sound_effect> in <Video_N>…` · Audio: `Reference the timbre in <Audio_N>…` | **Regenerates** (not your footage) |
| **Video editing** | "Make partial or global modifications based on the original video. **Parts not mentioned remain unchanged by default**" | Add: `<Element_Features> + <Timing> + <Location>` · Modify: `Strictly edit <Video_N>, modify <Original> to <New>` · Delete: name what to remove | **objects/attributes only — NOT camera/framing** |
| **Video extension** | "Continue the original video along the time dimension; style, subject, narrative remain consistent" | `Extend <Video_N> forward/backward to generate…` · Track-completion: `<Video_1> + transition + <Video_2> + …` | ≤3 clips, ≤15s total |
| **Image→video (first frame)** | "Specify the first frame image → content **visually coherent with** the image" | text + `first_frame` image | 1 image |
| **Image→video (first + last)** | "Specify starting & ending images → video that **smoothly connects** first and last frames, natural transition" | text (may include a camera move) + `first_frame` + `last_frame` | exactly **2 images** |
| **Text→video** | "highly random… source of inspiration" | description only | — |
| **Combined Tasks** | "**reference** one asset **+ edit** another" | `Reference [Dimension] of <Image/Video_N>, strictly edit <Video_X>, [Edits]` | reference + edit |

(2222480 L14-40; 2298881 L41-46, L97)

## Roles — what the model takes from each

| `role` | Contributes (quoted) | Used in |
|---|---|---|
| `reference_image` | "character image, visual style and screen composition" | multimodal reference |
| `reference_video` | "subject, camera movement, action performance and overall style" | multimodal reference |
| `reference_audio` | "timbre, music melody and dialogue content" | multimodal reference |
| `first_frame` | the **starting** frame; output coheres with it | image→video (first / first+last) |
| `last_frame` | the **ending** frame; output smoothly connects to it | image→video (first+last) |

(2291680 L44; 2298881 L42, L46)

**4 "functional roles" (asset-config strategy, 2222480 L117-124):** character anchoring
(appearance) · scene tone-setting (environment/style) · camera-movement reference (a
camera-move video) · rhythmic atmosphere (audio). Recommended **≤4-5 assets** total
(1-2 character + 1 scene + 1 camera-move video + 1 audio); "do not use the full asset
limit — too many assets → style conflicts, blurry subject ID, deviating results."

## Output parameters

| Param | Values / range |
|---|---|
| `resolution` | 480p, 720p, 1080p, 4k |
| `ratio` | 16:9, 4:3, 1:1, 3:4, 9:16, 21:9, adaptive |
| `duration` | **[4, 15]s or -1** (intelligent) — *2.0 series* |
| `frames` | **Seedance 1.0 only** (NOT 2.0) |
| `seed`, `camera_fixed`, `watermark` | listed params |

(2298881 L57, L70-71, L80, L83)

## Input limits

| Modality | Count | Per-item |
|---|---|---|
| Images | 0–9 (multimodal) | jpeg/png/webp/bmp/tiff/gif (+heic/heif on 2.0); AR 0.4–2.5; 300–6000px; <30 MB |
| Videos | 0–3, total ≤15s | single [2,15]s; mp4/mov; 24–60 fps; pixel area 409,600–8,295,044; <200 MB |
| Audio | 0–3, total ≤15s | single [2,15]s; wav/mp3; <15 MB |
| Request body | — | ≤64 MB; no base64 for large files |

(2298881 L143-169). Real human faces cannot be uploaded directly — must reuse trusted
Seedance outputs from the last 30 days (2291680 L78).

## The two "combine" concepts (do not confuse)

| "Combine" | Mixes | Status |
|---|---|---|
| **Combined Tasks** | reference task **+** edit task | ✅ documented (2222480 L36-40) |
| keyframes + references | `first_frame/last_frame` **+** `reference_*` roles in one request | ❌ **NOT shown** in any code sample; first/last-frame is a standalone 2-image task (2298881 L150-151). The only "first/last + reference" is the *indirect* prompt-described method (2291680 L53), loose/unconfirmed. |

## Behavioral rules

Official (doc-backed):
- **Edit/Extend verb routing:** refer to the clip as `@Video1` directly — do NOT write
  "reference @Video1" (re-routes to a reference task). (2222480 L35)
- **One camera move per shot:** "do not require push, pull, pan, move at the same time
  → increases image instability." (2222480 L91)
- **Lean:** "keep descriptions concise, avoid redundancy and semantic conflicts." (L54)
- **Decouple spatial vs temporal**; storyboard complex shots into separate shots. (L42)

`[RUNTIME]` (our tests — NOT in the docs, but consistent with the above):
- **A camera move is not an edit op.** Edit changes objects/attributes/frames, not
  framing. Adding a camera move needs *generation* (reference mode), which regenerates
  and drops existing effects → you **cannot** add a camera move while preserving a
  reference video's existing content/lighting in one Seedance pass. Do the move in post.
- **Multiple `reference_image` entries HARD-CUT** between the compositions (distinct
  semantic targets, not interpolation). Smooth transitions need `first_frame/last_frame`.
- **Stacking two big/temporal effects** (e.g. camera move + existing time-lapse) → the
  model keeps one, drops the other. One dominant change per shot.

---

## Footnote — where the docs & API notes live (for other agents)

**Official source docs** (BytePlus ModelArk, English):
- Prompt Guide — `https://docs.byteplus.com/en/docs/ModelArk/2222480`
- Model Reference (capabilities, code samples, edit/extend) — `…/ModelArk/2291680`
- Capabilities / params / input limits — `…/ModelArk/2298881`
- VideoPilot API suite — `…/ModelArk/2085689`

**Scrape tool:** `D:/tmp/bp_doc_extract.py` (no headless browser; extracts the
Quill-Delta JSON embedded in the page).
Usage: `python bp_doc_extract.py <doc_id> <out.md> ["Title"]`
Discover linked doc IDs: fetch a page and `grep -oE 'ModelArk/[0-9]{6,}'`.

**Local scraped copies (ephemeral — D:/tmp, re-scrape if missing):**
`bp_seedance_prompt_guide.md` (=2222480), `bp_2291680.md`, `bp_2298881.md`, `bp_2085689.md`.

**Our own notes / cross-refs:**
- `zerogen_comfy_custom/BEST_PRACTICES.md` §4a — official-vs-ours verification table.
- Auto-memory: `seedance_multi_ref_images_hard_cut`, `seedance_one_change_per_shot`,
  `seedance_reference_mode_framing`, `sd2_multi_video_refs_no_spatial_composite`.
- Prompt engine encoding these rules: `NV_Comfy_Utils/src/KNF_Utils/seedance_prompt_policy.py`
  (modes/skeletons/linter) + `NV_SeedanceShotV2` in `prompt_refiner.py`.

---

## Principles & best practices (consolidated 2026-06-29)

### Mental model
- **ONE generative endpoint that re-synthesizes the whole frame every time.** Reproduces
  what you reference, changes what you name; **never preserves source pixels**; reliably holds
  **one dominant change per shot**.
- **No `mode`/`task` API param** — behaviour is driven by **content roles + prompt verbs**.
  A node-level "mode" is a *client-side* label only.
- **Donor vs canvas:** `reference` = donor → new render; `edit` = canvas → re-render that
  *leans* on the source. Edit's "unmentioned stays unchanged" is a **soft bias**, not pixel
  preservation. Pixel-exact preservation → crop+stitch in our pipeline, not Seedance.
- **Subject-swap is the middle ground** — edit/reference converge (both regenerate). Default
  `edit` for max scene fidelity, `reference` to reinterpret the scene too.

### Prompt-craft
- **Verb routes the task (2222480 L35):** edit/extend use bare `@Video1`; never
  "reference @Video1". Reference uses "Reference … in @Video1 to generate".
- **LEAN wins (L54):** avoid redundancy / semantic conflicts; forceful absolutes make it rigid
  and passthrough-prone. (Our linter bans the CN strict-term `严格`; "strictly" is a *router*,
  keep the *change* wording lean.)
- **One dominant change per shot; one camera move per shot** (L91). Stacking two big/temporal
  effects → one drops. Decouple spatial vs temporal; storyboard into separate shots.
- **`@Image n` counts image_url items by ARRAY ORDER, not by role (2291680 L102)** → in
  keyframe modes the frames ARE `@Image1`/`@Image2` (so "from @Image1 to @Image2" bridge
  prompts are valid).
- **Multi-image identity (L142):** face/hair ← headshot, body/styling ← full-body. Headshot +
  full-body beats multi-view (multi-view causes ID drift).
- **Restyle needs an explicit style word (L165)** or it drifts back to realistic.

### Keyframe / greybox / extension
- **Keyframe modes are image-only** (no ref videos): first_frame = 1 image, first_last_frame = 2.
- **"White model" = greybox is an OFFICIAL term (2222480 L204)** — used to stabilize structure
  across extension/continuation (strip the color/texture layer where drift accumulates). Not a
  documented *capture* method.
- **Extension:** activate with the verb `Extend @Video1 forward/backward`. `duration` = the
  *generated* clip length [4-15] (so `auto_from_ref` makes the extension = input length — it is
  the extension length, NOT append). The **seam/"jump" is documented**: (A) extension
  degradation, mottled face color, compounds across continuations (L199-207 → white-model + HD
  refs + fewer continuations); (B) input/output res-AR mismatch → frame jumps (2291680 L136 →
  match res + ratio). Deeper: Seedance regenerates → inherent color seam → color-match the seam
  (CropColorFix / `apply_lab_lowfreq`) or a short low-denoise blend.

### Gotchas / safety
- **Real faces can't be uploaded (2.0)** → authorized-real-person registration / trusted
  ≤30-day ModelArk outputs / preset avatars / strip-the-face.
- **OUTPUT content gate** (post-gen likeness/copyright) is separate from the input gate and
  **not** asset-bypassable.
- **Image auto-crop = center-crop** on AR mismatch (chops tall full-body refs) → frame with
  safe margins or match ratio.
- **Webhook (`callback_url`) exists** but needs a public endpoint; **concurrency cap
  auto-queues** the overflow (Enterprise = 10, Individual = 3; 4K = 1).
- **Multimodal refs blend SEMANTICALLY, not spatially** — can't composite layers; do
  compositing in our pipeline.

### VP / real-shoot capture (academic; API gates aside)
Capture what's expensive to *synthesize* (real light transport, micro-detail, subject↔scene
interaction); avoid baking what's expensive to *reverse* (baked light, spill, fixed BG). LED
wall = unfakeable interactive light; greenscreen = flexibility but needs relight; greybox =
environment anchor (geometry/parallax) → photorealize BG + composite the real subject (plate
authority). Full matrix: `seedance_api_docs/VP_CAPTURE_ACADEMIC_MATRIX.md`.
