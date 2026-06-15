# Vendored files

These files are **copied** (vendored) from `NV_Comfy_Utils/src/KNF_Utils/` so this
pack is self-contained. They are also used by NV_Comfy_Utils' Moyu / Comfy-proxy
Seedance paths, so the same logic lives in both repos and can **drift**.

| Vendored file | Canonical source (NV_Comfy_Utils) |
|---|---|
| `src/zerogen_utils/api_keys.py` | `src/KNF_Utils/api_keys.py` |
| `src/zerogen_utils/nv_seedance_upload_utils.py` | `src/KNF_Utils/nv_seedance_upload_utils.py` |
| `src/zerogen_utils/seedance_chunked_loop_ops.py` | `src/KNF_Utils/seedance_chunked_loop_ops.py` |

`src/zerogen_utils/seedance_types.py` is **not** vendored — it is a tiny extraction
of the `SEEDANCE_UPLOAD_CONFIG` IO-type (matched across packs by its string name),
not subject to drift.

## Keeping them in sync

NV_Comfy_Utils is the canonical owner. When you change one of the files above in
either repo, port the change to the other and verify byte-identity:

```bash
python tools/check_vendor_sync.py --nv /path/to/NV_Comfy_Utils
```

Exit code 0 = in sync; non-zero = drift (prints which files differ).
