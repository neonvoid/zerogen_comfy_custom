"""Shared ComfyUI IO socket type(s) for the native Seedance pipeline.

`SEEDANCE_UPLOAD_CONFIG` is a custom typed socket carrying the preprocessed +
uploaded Seedance config between the prep node and the native caller. ComfyUI
matches socket types by their string name, so this definition is wire-compatible
with any other pack that declares a socket named "SEEDANCE_UPLOAD_CONFIG"
(e.g. NV_Comfy_Utils' comfy-proxy seedance_prep). Extracted here so the native
pack is self-contained without vendoring the proxy prep node.
"""

from comfy_api.latest._io import Custom as _IOCustom

SEEDANCE_UPLOAD_CONFIG = _IOCustom("SEEDANCE_UPLOAD_CONFIG")
