"""Offline unit tests for nv_byteplus_seedance_gen._poll_task.

The dev clone has no torch / comfy_api (runtime lives on a separate machine), so
we stub the heavy module-level imports, then drive the poll loop with a fake
_get_task. Validates the diagnostic fixes:
  - unknown status is surfaced immediately (no silent poll-to-deadline)
  - a 429 storm does NOT burn the consecutive-failure budget (rate limit != outage)
  - terminal statuses are returned for the caller to classify
  - the monotonic deadline still fires

Run: python tests/test_seedance_poll.py
"""
import asyncio
import importlib.util
import os
import sys
import types

# --- Stub heavy deps the poll logic doesn't actually exercise -----------------
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _FakeIO:
    """Stand-in for comfy_api.latest.IO. Only IO.ComfyNode (a base class) is touched
    at import; everything else (Schema/Float/... ) is reached lazily inside
    define_schema(), which the poll tests never call."""

    class ComfyNode:
        pass

    def __getattr__(self, name):
        return type(name, (), {})


_stub("torch", zeros=lambda *a, **k: None)
_stub("comfy_api")
_stub("comfy_api.latest", IO=_FakeIO())
_stub("comfy")
_stub("comfy.model_management",
      throw_exception_if_processing_interrupted=lambda: None)

# Synthetic package so the module's `from .api_keys import resolve_api_key` resolves.
pkg = types.ModuleType("zerogen_utils")
pkg.__path__ = [os.path.abspath(os.path.join(_SRC, "zerogen_utils"))]
sys.modules["zerogen_utils"] = pkg
_stub("zerogen_utils.api_keys", resolve_api_key=lambda *a, **k: "test-key")

_spec = importlib.util.spec_from_file_location(
    "zerogen_utils.nv_byteplus_seedance_gen",
    os.path.join(_SRC, "zerogen_utils", "nv_byteplus_seedance_gen.py"),
)
g = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = g
_spec.loader.exec_module(g)

# Make all sleeps instant so a 30s rate-limit backoff doesn't stall the test.
async def _fast_sleep(_):  # noqa: ANN001
    return None


g.asyncio.sleep = _fast_sleep

_CLS = object()  # _emit_poll_progress swallows everything; a bare object is fine.


def _scripted_get_task(script):
    """Return a fake _get_task that yields/raises items from `script` in order.
    Each item is either a dict (returned) or an Exception instance (raised)."""
    it = iter(script)

    async def fake(_session, _key, _task_id):
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    return fake


def _run(script, *, interval=0.0, timeout=30.0):
    g._get_task = _scripted_get_task(script)
    return asyncio.run(_poll(interval, timeout))


async def _poll(interval, timeout):
    return await g._poll_task(_CLS, None, "k", "task-123", interval, timeout)


# --- Tests --------------------------------------------------------------------

def test_happy_path_returns_succeeded():
    resp = _run([
        {"status": "queued"},
        {"status": "running"},
        {"status": "succeeded", "content": {"video_url": "x"}},
    ])
    assert resp["status"] == "succeeded", resp


def test_unknown_status_raises_immediately():
    try:
        _run([{"status": "running"}, {"status": "frobnicated"}])
    except RuntimeError as e:
        assert "unrecognized" in str(e).lower(), str(e)
        assert "frobnicated" in str(e), str(e)
    else:
        raise AssertionError("unknown status did not raise")


def test_terminal_failed_is_returned_for_caller():
    # _poll_task returns terminal failures; _generate_one is what raises on them.
    resp = _run([{"status": "running"}, {"status": "failed", "error": {"code": "X"}}])
    assert resp["status"] == "failed", resp


def test_rate_limit_storm_does_not_burn_failure_budget():
    # 15 consecutive 429s (> _MAX_CONSECUTIVE_POLL_FAILURES=10) must NOT raise the
    # "outage" error — a rate limit is expected backpressure. Then it succeeds.
    storm = [g._TransientPollHTTPError("429", rate_limited=True) for _ in range(15)]
    resp = _run(storm + [{"status": "succeeded"}], timeout=600.0)
    assert resp["status"] == "succeeded", resp


def test_real_transient_storm_does_raise_outage():
    # Non-rate-limit transient errors DO burn the budget and raise after the cap.
    storm = [g._TransientPollHTTPError("503") for _ in range(g._MAX_CONSECUTIVE_POLL_FAILURES)]
    try:
        _run(storm + [{"status": "succeeded"}], timeout=600.0)
    except RuntimeError as e:
        assert "consecutively" in str(e).lower(), str(e)
    else:
        raise AssertionError("transient storm did not raise the outage error")


def test_deadline_fires_on_perpetual_running():
    # timeout=0 → the monotonic deadline check at the top trips on a running task.
    try:
        _run([{"status": "running"}] * 50, timeout=0.0)
    except RuntimeError as e:
        assert "timed out" in str(e).lower(), str(e)
    else:
        raise AssertionError("perpetual running did not time out")


def test_multi_video_tag_injection():
    # 2 videos + 1 image, empty-ish prompt → @Video1 @Video2 @Image1 prefixed.
    out = g._auto_inject_tags("dance", n_images=1, n_videos=2)
    assert out.startswith("@Video1 @Video2 @Image1 "), out
    assert out.endswith("dance"), out


def test_tag_injection_bool_backcompat():
    # Legacy callers passed has_video=True positionally → must still mean 1 video.
    assert g._auto_inject_tags("x", 0, True).startswith("@Video1 "), "bool True should mean 1 video"
    assert g._auto_inject_tags("x", 0, False) == "x", "bool False should inject no video tag"


def test_tag_injection_respects_existing_tags():
    # User already wrote @Video tags → don't double-inject.
    p = "@Video1 swaps with @Video2"
    assert g._auto_inject_tags(p, n_images=0, n_videos=2) == p


def test_build_content_multiple_videos():
    content = g._build_content("p", ["asset://img"], ["asset://v1", "https://x/v2.mp4", "  "])
    vids = [c for c in content if c["type"] == "video_url"]
    imgs = [c for c in content if c["type"] == "image_url"]
    assert len(vids) == 2, vids  # blank entry skipped
    assert all(c["role"] == "reference_video" for c in vids)
    assert vids[0]["video_url"]["url"] == "asset://v1"
    assert len(imgs) == 1


def test_parse_ref_urls_video_multiline_and_exclusion():
    assert g._parse_ref_urls("a\nb\nc", "", kind="video") == ["a", "b", "c"]
    assert g._parse_ref_urls("", "solo", kind="video") == ["solo"]
    try:
        g._parse_ref_urls("a", "b", kind="video")
    except ValueError as e:
        assert "video" in str(e).lower(), str(e)
    else:
        raise AssertionError("multiline+singular should raise")


def test_parse_retry_after():
    assert g._parse_retry_after("12") == 12.0
    assert g._parse_retry_after("  7.5 ") == 7.5
    assert g._parse_retry_after(None) is None
    assert g._parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None  # HTTP-date → None
    assert g._parse_retry_after("-3") == 0.0  # clamped non-negative


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
