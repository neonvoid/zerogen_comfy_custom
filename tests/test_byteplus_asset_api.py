"""Unit tests for nv_byteplus_asset_api + nv_byteplus_asset_cache.

Fixture-free plain functions so the standalone runner
(`run_byteplus_asset_api_tests.py`) can execute them without pytest collection
— the dev-clone can't import the full KNF_Utils package chain (heartbeat →
server), so pytest's package-walk collection is unusable here (known
two-machine-setup limitation). pytest can still collect this file on an
environment where the chain imports.

The V4 signing test pins the module against an independent reference
implementation transplanted verbatim from the scratch probe that was
runtime-validated against the live BytePlus API on 2026-06-11
(D:/tmp/byteplus_asset_smoke.py — ListAssetGroups/ListAssets/CreateAsset/
GetAsset all returned HTTP 200 with these exact signing semantics).
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types
from types import SimpleNamespace

import torch

# ---------------------------------------------------------------------------
# Load the modules directly (bypass the heavy KNF_Utils/__init__ chain, which
# imports `server` and only resolves inside a live ComfyUI process). Synthetic
# package so the API module's relative `from .api_keys import ...` works.
# ---------------------------------------------------------------------------

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_SRC_DIR = _TESTS_DIR.parent / "src" / "zerogen_utils"

_PKG_NAME = "_bp_test_pkg"
if _PKG_NAME not in sys.modules:
    _pkg = types.ModuleType(_PKG_NAME)
    _pkg.__path__ = [str(_SRC_DIR)]
    sys.modules[_PKG_NAME] = _pkg


def _load(submodule: str):
    name = f"{_PKG_NAME}.{submodule}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(_SRC_DIR / f"{submodule}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


api_keys_mod = _load("api_keys")
bp_api = _load("nv_byteplus_asset_api")
bp_cache = _load("nv_byteplus_asset_cache")


# ---------------------------------------------------------------------------
# Tiny fixture substitutes (no pytest dependency)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _env(**kv):
    """Temporarily set (value) / delete (None) environment variables."""
    saved = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _temp_cache():
    """Point the byteplus cache at a throwaway file; yield its Path."""
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "byteplus_assets.json"
        with _env(NV_BYTEPLUS_ASSET_CACHE_PATH=str(path)):
            yield path


@contextlib.contextmanager
def _attr(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


def _raises(exc_type, fn, match=None):
    try:
        fn()
    except exc_type as e:
        if match is not None:
            assert match in str(e), f"expected {match!r} in {e}"
        return
    raise AssertionError(f"expected {exc_type.__name__} was not raised")


# =============================================================================
# V4 signing — pinned against the live-validated reference implementation
# =============================================================================


def _reference_signature(action, payload, ak, sk, host, x_date, region, service, version):
    """Verbatim signing math from the runtime-validated probe (2026-06-11)."""
    short_date = x_date[:8]
    content_sha = hashlib.sha256(payload).hexdigest()
    query = f"Action={action}&Version={version}"
    canonical_headers = (
        f"content-type:application/json\n"
        f"host:{host}\n"
        f"x-content-sha256:{content_sha}\n"
        f"x-date:{x_date}\n"
    )
    signed_headers = "content-type;host;x-content-sha256;x-date"
    canonical_request = "\n".join(["POST", "/", query, canonical_headers, signed_headers, content_sha])
    scope = f"{short_date}/{region}/{service}/request"
    string_to_sign = "\n".join(["HMAC-SHA256", x_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest()])

    def _h(key, msg):
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    k_signing = _h(_h(_h(_h(sk.encode(), short_date), region), service), "request")
    return hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest(), scope, content_sha


def test_v4_signature_matches_validated_reference():
    payload = json.dumps({"Filter": {"GroupType": "AIGC"}, "PageNumber": 1, "PageSize": 20}).encode()
    ak, sk = "AKLTtestaccesskey0000", "testsecretkey=="
    host = "ark.ap-southeast-1.byteplusapi.com"
    x_date = "20260611T204700Z"

    headers = bp_api.build_v4_signed_headers(
        action="ListAssetGroups", payload=payload, access_key=ak, secret_key=sk, host=host, x_date=x_date,
    )
    ref_sig, ref_scope, ref_sha = _reference_signature(
        "ListAssetGroups", payload, ak, sk, host, x_date, bp_api.ARK_REGION, bp_api.ARK_SERVICE, bp_api.ARK_ACTION_VERSION,
    )

    assert headers["X-Date"] == x_date
    assert headers["X-Content-Sha256"] == ref_sha
    assert headers["Host"] == host
    assert headers["Content-Type"] == "application/json"
    auth = headers["Authorization"]
    assert auth.startswith("HMAC-SHA256 ")
    assert f"Credential={ak}/{ref_scope}" in auth
    assert "SignedHeaders=content-type;host;x-content-sha256;x-date" in auth
    assert auth.endswith(f"Signature={ref_sig}")


def test_v4_signature_changes_with_payload_and_date():
    common = dict(action="GetAsset", access_key="ak", secret_key="sk", host="h.example.com")
    a = bp_api.build_v4_signed_headers(payload=b"{}", x_date="20260611T000000Z", **common)
    b = bp_api.build_v4_signed_headers(payload=b'{"Id":"x"}', x_date="20260611T000000Z", **common)
    c = bp_api.build_v4_signed_headers(payload=b"{}", x_date="20260612T000000Z", **common)
    sigs = {h["Authorization"].rsplit("Signature=", 1)[1] for h in (a, b, c)}
    assert len(sigs) == 3, "payload and date must both perturb the signature"


# =============================================================================
# Small pure helpers
# =============================================================================


def test_deterministic_asset_name_is_stable_and_short():
    h = "ab" * 32
    # 24 hex chars (96 bits) per R0 review — collision-resistant remote-adopt key
    assert bp_api.deterministic_asset_name(h) == "nv-ref-" + "ab" * 12
    assert bp_api.deterministic_asset_name(h) == bp_api.deterministic_asset_name(h)
    assert len(bp_api.deterministic_asset_name(h)) <= 64  # API Name cap


def test_looks_like_group_id():
    cases = [
        ("group-20260612043243-zrpsr", True),
        ("  group-x  ", True),
        ("GROUP-UPPER", True),
        ("jon_1", False),
        ("nv_refs", False),
        ("", False),
        ("asset-20260612045305-82f5h", False),
    ]
    for ref, expected in cases:
        assert bp_api.looks_like_group_id(ref) is expected, f"looks_like_group_id({ref!r})"


# =============================================================================
# Image preflight — native constraints (300-6000px sides, W/H aspect (0.4, 2.5))
# =============================================================================


def _img(h, w):
    return torch.zeros((1, h, w, 3))


def test_preflight_image_accepts_typical_refs():
    ok, msg = bp_api.preflight_image(_img(2048, 2048))
    assert ok, msg
    ok, msg = bp_api.preflight_image(_img(1920, 1080))  # portrait 0.5625 aspect
    assert ok, msg


def test_preflight_image_rejects_side_violations():
    for h, w in [(299, 1000), (1000, 299), (6001, 1000), (1000, 6001)]:
        ok, msg = bp_api.preflight_image(_img(h, w))
        assert not ok, f"({h},{w}) should fail"
        assert "outside native range" in msg


def test_preflight_image_rejects_extreme_aspect():
    ok, msg = bp_api.preflight_image(_img(3000, 1000))  # W/H = 0.333 < 0.4
    assert not ok and "aspect" in msg
    ok, msg = bp_api.preflight_image(_img(1000, 3000))  # W/H = 3.0 > 2.5
    assert not ok and "aspect" in msg


# =============================================================================
# Video preflight — THE key delta vs the Moyu plane: 1080p fits natively
# =============================================================================


class _FakeVideo:
    def __init__(self, frames, h, w, fps, duration=None):
        self._images = torch.zeros((frames, h, w, 3))
        self._fps = fps
        self._duration = duration if duration is not None else frames / fps

    def get_components(self):
        return SimpleNamespace(images=self._images, frame_rate=self._fps)

    def get_duration(self):
        return self._duration


def test_preflight_video_accepts_1080p():
    ok, msg = bp_api.preflight_video(_FakeVideo(frames=120, h=1080, w=1920, fps=24.0))
    assert ok, f"1080p (2,073,600 px < {bp_api.VIDEO_MAX_PIXEL_AREA:,}) must pass the NATIVE preflight: {msg}"


def test_preflight_video_accepts_ntsc_24p():
    ok, msg = bp_api.preflight_video(_FakeVideo(frames=120, h=720, w=1280, fps=23.976))
    assert ok, msg


def test_preflight_video_rejects_1440p_area():
    ok, msg = bp_api.preflight_video(_FakeVideo(frames=120, h=1440, w=2560, fps=24.0))
    assert not ok and "pixel area" in msg and "TOO LARGE" in msg


def test_preflight_video_rejects_bad_fps():
    for fps in [10.0, 22.9, 61.0, 120.0]:
        ok, msg = bp_api.preflight_video(_FakeVideo(frames=120, h=720, w=1280, fps=fps))
        assert not ok and "fps" in msg, f"fps={fps} should fail"


def test_preflight_video_rejects_bad_duration():
    for duration in [1.0, 16.0, 60.0]:
        ok, msg = bp_api.preflight_video(_FakeVideo(frames=120, h=720, w=1280, fps=24.0, duration=duration))
        assert not ok and "duration" in msg, f"duration={duration} should fail"


def test_preflight_video_collects_multiple_violations():
    ok, msg = bp_api.preflight_video(_FakeVideo(frames=10, h=1440, w=2560, fps=10.0, duration=1.0))
    assert not ok
    assert "duration" in msg and "fps" in msg and "pixel area" in msg


def test_preflight_video_uninspectable_fails_closed():
    class Broken:
        def get_components(self):
            raise RuntimeError("no components")

    ok, msg = bp_api.preflight_video(Broken())
    assert not ok and "could not inspect" in msg


# =============================================================================
# Credential resolution
# =============================================================================


def test_resolve_ark_ak_sk_explicit_wins():
    with _env(ARK_ACCESS_KEY="env-ak", ARK_SECRET_KEY="env-sk"):
        assert bp_api.resolve_ark_ak_sk(" explicit-ak ", "explicit-sk") == ("explicit-ak", "explicit-sk")


def test_resolve_ark_ak_sk_env_fallback():
    with _env(ARK_ACCESS_KEY="env-ak", ARK_SECRET_KEY="env-sk"):
        assert bp_api.resolve_ark_ak_sk("", "") == ("env-ak", "env-sk")


def test_resolve_ark_ak_sk_missing_raises():
    # Block the real .env (which may carry live keys on dev machines) from
    # satisfying the lookup; asserts the missing-credential error path.
    with _attr(api_keys_mod, "_ENV_LOADED", True), _env(ARK_ACCESS_KEY=None, ARK_SECRET_KEY=None):
        _raises(RuntimeError, lambda: bp_api.resolve_ark_ak_sk("", ""), match="ARK_ACCESS_KEY")


def test_resolve_ark_ak_sk_half_pair_raises():
    with _attr(api_keys_mod, "_ENV_LOADED", True), _env(ARK_ACCESS_KEY="env-ak", ARK_SECRET_KEY=None):
        _raises(RuntimeError, lambda: bp_api.resolve_ark_ak_sk("", ""))


# =============================================================================
# Cache — isolated to a temp file via NV_BYTEPLUS_ASSET_CACHE_PATH
# =============================================================================


def test_cache_roundtrip():
    with _temp_cache() as cache_file:
        assert bp_cache.lookup("AKLTabcdefgh1234", "h" * 64) is None
        entry = bp_cache.register(
            "AKLTabcdefgh1234", "h" * 64,
            asset_id="asset-x", asset_url="asset://asset-x", asset_type="Image",
            tag="t", group_id="group-y", group_name="jon_1", source_url_tail="https://x/y.png",
        )
        assert entry["asset_id"] == "asset-x"
        got = bp_cache.lookup("AKLTabcdefgh1234", "h" * 64)
        assert got is not None
        assert got["asset_url"] == "asset://asset-x"
        assert got["group_id"] == "group-y"
        assert got["verified_at"] == got["registered_at"]
        assert cache_file.is_file()


def test_cache_namespaced_by_access_key():
    with _temp_cache():
        bp_cache.register("AKLTaaaaaaaa1111", "h" * 64, asset_id="a1", asset_url="asset://a1", asset_type="Image")
        assert bp_cache.lookup("AKLTbbbbbbbb2222", "h" * 64) is None
        assert bp_cache.lookup("AKLTaaaaaaaa1111", "h" * 64)["asset_id"] == "a1"


def test_cache_invalidate_self_heal_path():
    with _temp_cache():
        bp_cache.register("ak-0123456789ab", "h" * 64, asset_id="dead", asset_url="asset://dead", asset_type="Image")
        assert bp_cache.invalidate("ak-0123456789ab", "h" * 64) is True
        assert bp_cache.lookup("ak-0123456789ab", "h" * 64) is None
        assert bp_cache.invalidate("ak-0123456789ab", "h" * 64) is False


def test_cache_mark_verified_updates_timestamp():
    with _temp_cache():
        bp_cache.register("ak-0123456789ab", "h" * 64, asset_id="a", asset_url="asset://a", asset_type="Video")
        with _attr(bp_cache.time, "time", lambda: 9_999_999_999):
            bp_cache.mark_verified("ak-0123456789ab", "h" * 64)
        assert bp_cache.lookup("ak-0123456789ab", "h" * 64)["verified_at"] == 9_999_999_999


def test_cache_corrupt_file_degrades_gracefully():
    with _temp_cache() as cache_file:
        cache_file.write_text("{not json", encoding="utf-8")
        assert bp_cache.lookup("ak-0123456789ab", "h" * 64) is None
        bp_cache.register("ak-0123456789ab", "h" * 64, asset_id="a", asset_url="asset://a", asset_type="Image")
        assert bp_cache.lookup("ak-0123456789ab", "h" * 64)["asset_id"] == "a"


def test_cache_hash_image_tensor_stable():
    t = torch.rand((1, 8, 8, 3))
    assert bp_cache.hash_image_tensor(t) == bp_cache.hash_image_tensor(t.clone())
    assert bp_cache.hash_image_tensor(t) != bp_cache.hash_image_tensor(torch.rand((1, 8, 8, 3)))


def test_cache_hash_video_signature():
    v1 = _FakeVideo(frames=10, h=16, w=16, fps=24.0)
    v2 = _FakeVideo(frames=10, h=16, w=16, fps=24.0)
    # zeros frames → identical first/last/count signature
    assert bp_cache.hash_video_input(v1) == bp_cache.hash_video_input(v2)
    v3 = _FakeVideo(frames=11, h=16, w=16, fps=24.0)
    assert bp_cache.hash_video_input(v1) != bp_cache.hash_video_input(v3)


def test_image_max_bytes_constant_sane():
    assert bp_api.IMAGE_MAX_BYTES == 30 * 1024 * 1024
