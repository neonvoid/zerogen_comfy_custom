# zerogen_comfy_custom

Native **BytePlus / Volcengine Ark Seedance 2.0** ComfyUI nodes — the ark-direct
pipeline (asset library + native generation) as a standalone pack.

## What's in here

| Node | Purpose |
|---|---|
| `Zerogen_ByteplusImageAssetRegister` / `Zerogen_ByteplusVideoAssetRegister` / `Zerogen_ByteplusImageBatchRegister` | Register assets into the BytePlus ModelArk trusted asset library (V4-signed) |
| `Zerogen_ByteplusSeedanceGen` | Native generation on `ark.ap-southeast.bytepluses.com` (dreamina-) |
| `Zerogen_ByteplusSeedanceJobConfig` / `Zerogen_ByteplusSeedanceMultiJob` | Single-shot parallel multi-job fanout |
| `Zerogen_SeedanceNativeRefVideo` / `_V2` | Native generation on `ark.cn-beijing.volces.com` |
| `Zerogen_SeedanceNativeChunkedLoop_V2` | Multi-chunk native generation |
| `Zerogen_SeedancePrep_V2` | Tensor-in preprocessing + upload (emits the upload config) |
| `Zerogen_SeedanceFetchTask` | Retrieve async tasks by id |

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/neonvoid/zerogen_comfy_custom.git
pip install -r zerogen_comfy_custom/requirements.txt   # aiohttp, boto3 (torch/numpy/Pillow come with ComfyUI)
cp zerogen_comfy_custom/.env.example zerogen_comfy_custom/.env   # then fill in credentials
```

Credentials: copy [.env.example](.env.example) → `.env` and fill it in — `ARK_API_KEY`
for generation, Access Key (`ARK_ACCESS_KEY`/`ARK_SECRET_KEY`) for the asset library,
optional B2 creds for staging. Restart ComfyUI after installing.

## Docs

- **[BEST_PRACTICES.md](BEST_PRACTICES.md)** — intro, setup, the asset→generation flow,
  SD2 prompting + reference-video best practices, model limits, troubleshooting.
