"""Standalone runner for the BytePlus asset API/cache unit tests.

Run from the tests/ directory:
    python run_byteplus_asset_api_tests.py

Exists because the dev clone can't import the full KNF_Utils package chain
(heartbeat -> server), which breaks pytest's package-walk collection. The
test module is self-contained (loads its targets via importlib) and
fixture-free, so plain function invocation is sufficient.
"""

import sys
import traceback

sys.path.insert(0, ".")

import test_byteplus_asset_api as t


def main():
    tests = [(name, fn) for name, fn in vars(t).items() if name.startswith("test_") and callable(fn)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception:
            print(f"FAIL: {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed (of {len(tests)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
