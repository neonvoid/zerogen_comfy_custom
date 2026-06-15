# zerogen_comfy_custom

Native **BytePlus / Volcengine Ark Seedance 2.0** ComfyUI nodes — the ark-direct
pipeline, extracted from `NV_Comfy_Utils` into a standalone, shareable pack.

## What's in here

| Node | Module | Purpose |
|---|---|---|
| `NV_ByteplusImageAssetRegister` / `NV_ByteplusVideoAssetRegister` / `NV_ByteplusImageBatchRegister` | `nv_byteplus_asset_register` | Register assets into the BytePlus ModelArk trusted asset library (V4-signed) |
| `NV_ByteplusSeedanceGen` | `nv_byteplus_seedance_gen` | Native generation on `ark.ap-southeast.bytepluses.com` (dreamina-) |
| `NV_ByteplusSeedanceJobConfig` / `NV_ByteplusSeedanceMultiJob` | `nv_byteplus_seedance_multijob` | Single-shot parallel multi-job fanout |
| `NV_SeedanceNativeRefVideo` / `_V2` | `nv_seedance_native` / `_v2` | Native generation on `ark.cn-beijing.volces.com` |
| `NV_SeedanceNativeChunkedLoop_V2` | `seedance_native_chunked_loop` | Multi-chunk native generation |
| `NV_SeedancePrep_V2` | `nv_seedance_prep_v2` | Tensor-in preprocessing + upload (emits the upload config) |
| `NV_SeedanceFetchTask` | `nv_seedance_fetch_task` | Retrieve async tasks by id |

## Install

Clone into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/neonvoid/zerogen_comfy_custom.git
pip install -r zerogen_comfy_custom/requirements.txt   # aiohttp, boto3 (torch/numpy/Pillow come with ComfyUI)
cp zerogen_comfy_custom/.env.example zerogen_comfy_custom/.env   # then fill in credentials
```

Credentials: copy [.env.example](.env.example) → `.env` and fill it in — `ARK_API_KEY`
for generation, Access Key (`ARK_ACCESS_KEY`/`ARK_SECRET_KEY`) for the asset library,
optional B2 creds for staging.

## Docs

- **[BEST_PRACTICES.md](BEST_PRACTICES.md)** — intro, setup, the asset→generation flow,
  SD2 prompting + reference-video best practices, model limits, troubleshooting.
- **[VENDOR.md](VENDOR.md)** — vendored-file list + drift-sync policy.

## Relationship to NV_Comfy_Utils

- **Here:** only the native ark-direct Seedance pipeline.
- **Stays in NV_Comfy_Utils:** general LLM/VLM nodes (incl. the Seedance prompt
  tools), the Comfy-proxy Seedance path, and all Moyu nodes.
- A few infra files are **vendored** (copied) from NV_Comfy_Utils so this pack is
  self-contained: `api_keys.py`, `nv_seedance_upload_utils.py`,
  `seedance_chunked_loop_ops.py`. See [VENDOR.md](VENDOR.md) and run
  `python tools/check_vendor_sync.py` to detect drift.

Node ids are unchanged from NV_Comfy_Utils, so existing workflows keep working
once this pack replaces the native nodes (which were removed from NV_Comfy_Utils).
