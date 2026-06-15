"""
Shared API key resolver for NV_Comfy_Utils nodes.

Resolves API keys with this precedence:
  1. Explicit value from node input (override)
  2. Environment variable (GEMINI_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY)
  3. .env file in the NV_Comfy_Utils root directory

Usage:
    from .api_keys import resolve_api_key

    key = resolve_api_key(api_key_input, provider="gemini")
"""

import os

# ---------------------------------------------------------------------------
# .env loader (minimal, no python-dotenv dependency)
# ---------------------------------------------------------------------------

_ENV_LOADED = False


def _load_dotenv_once():
    """Load .env file from NV_Comfy_Utils root, once per process."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    # NV_Comfy_Utils root is three levels up from this file:
    #   src/KNF_Utils/api_keys.py -> src/KNF_Utils/ -> src/ -> NV_Comfy_Utils/
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(package_root, ".env")

    if not os.path.isfile(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Strip surrounding quotes
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                # Only set if not already in environment (real env vars take priority)
                if key and key not in os.environ:
                    os.environ[key] = value
        print(f"[NV_Comfy_Utils] Loaded .env from {env_path}")
    except Exception as e:
        print(f"[NV_Comfy_Utils] Warning: failed to read .env: {e}")


# ---------------------------------------------------------------------------
# Provider -> env var mapping
# ---------------------------------------------------------------------------

_PROVIDER_ENV_VARS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "volcengine": ("VOLCENGINE_ARK_API_KEY", "ARK_API_KEY"),
    "moyu": ("MOYU_API_KEY",),
    # B2 uses TWO credentials (key id + application key) — resolve via
    # resolve_b2_credentials() below, not resolve_api_key(). Listed here
    # for env-var enumeration / diagnostic completeness.
    "b2": ("B2_KEY_ID", "B2_APPLICATION_KEY"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_api_key(api_key: str, provider: str = "gemini") -> str:
    """Resolve API key from explicit input, environment, or .env file.

    Args:
        api_key: Explicit key from node input (takes priority). Empty string = skip.
        provider: "gemini" or "openrouter" — determines which env vars to check.

    Returns:
        Resolved API key string.

    Raises:
        RuntimeError: If no key is found from any source.
    """
    # 1. Explicit input wins
    if api_key and api_key.strip():
        return api_key.strip()

    # 2. Load .env (if present and not already loaded)
    _load_dotenv_once()

    # 3. Check environment variables
    env_vars = _PROVIDER_ENV_VARS.get(provider, ())
    for var in env_vars:
        val = os.environ.get(var)
        if val and val.strip():
            print(f"[NV_Comfy_Utils] API key loaded from {var}")
            return val.strip()

    # 4. No key found
    var_names = " / ".join(env_vars) if env_vars else provider.upper() + "_API_KEY"
    raise RuntimeError(
        f"No API key for {provider}. Either:\n"
        f"  - Set {var_names} environment variable, or\n"
        f"  - Add it to NV_Comfy_Utils/.env file, or\n"
        f"  - Paste it into the api_key node input."
    )


def resolve_b2_credentials(
    key_id: str = "",
    application_key: str = "",
    bucket: str = "",
    region: str = "",
) -> tuple[str, str, str, str]:
    """Resolve B2 credentials + bucket + region.

    B2 uses two-part credentials (key_id + application_key), so it doesn't fit
    the single-string resolve_api_key contract. Returns
    (key_id, application_key, bucket, region) with the same precedence as
    resolve_api_key: explicit input > environment > .env file.

    Defaults: bucket=nv-comfy-moyu, region=us-east-005. Override via the
    B2_BUCKET / B2_REGION env vars or by passing values explicitly.

    Raises RuntimeError if key_id or application_key cannot be resolved.
    """
    # 1. Explicit inputs win
    resolved_id = (key_id or "").strip()
    resolved_key = (application_key or "").strip()
    resolved_bucket = (bucket or "").strip()
    resolved_region = (region or "").strip()

    # 2. Fall back to env / .env file for anything still empty
    if not resolved_id or not resolved_key or not resolved_bucket or not resolved_region:
        _load_dotenv_once()
        if not resolved_id:
            resolved_id = (os.environ.get("B2_KEY_ID") or "").strip()
        if not resolved_key:
            resolved_key = (os.environ.get("B2_APPLICATION_KEY") or "").strip()
        if not resolved_bucket:
            resolved_bucket = (os.environ.get("B2_BUCKET") or "nv-comfy-moyu").strip()
        if not resolved_region:
            resolved_region = (os.environ.get("B2_REGION") or "us-east-005").strip()

    if not resolved_id or not resolved_key:
        raise RuntimeError(
            "No B2 credentials. Either:\n"
            "  - Set B2_KEY_ID and B2_APPLICATION_KEY environment variables, or\n"
            "  - Add them to NV_Comfy_Utils/.env file, or\n"
            "  - Paste them into the node's b2_key_id / b2_application_key inputs."
        )

    return resolved_id, resolved_key, resolved_bucket, resolved_region
