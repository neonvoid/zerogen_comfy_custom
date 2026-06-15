"""zerogen_comfy_custom — native BytePlus / Volcengine Ark Seedance 2.0 ComfyUI nodes.

Standalone, shareable pack extracted from NV_Comfy_Utils. Contains only the native
ark-direct Seedance pipeline (asset library register + native generation + chunked
loop + prep). LLM/VLM prompt tooling and the Comfy-proxy / Moyu Seedance paths
remain in NV_Comfy_Utils.
"""

from .src.zerogen_utils import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
