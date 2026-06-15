#!/usr/bin/env python3
"""Detect drift between zerogen's vendored files and their canonical NV_Comfy_Utils source.

Usage:
    python tools/check_vendor_sync.py --nv /path/to/NV_Comfy_Utils

Exit 0 = all vendored files byte-identical to source. Exit 1 = drift (or missing).
See VENDOR.md for the file list and sync policy.
"""
import argparse
import hashlib
import pathlib
import sys

VENDORED = [
    "api_keys.py",
    "nv_seedance_upload_utils.py",
    "seedance_chunked_loop_ops.py",
]


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nv", required=True, help="Path to the NV_Comfy_Utils repo root")
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent.parent / "src" / "zerogen_utils"
    nv = pathlib.Path(args.nv).resolve() / "src" / "KNF_Utils"

    drift = []
    for name in VENDORED:
        z, s = here / name, nv / name
        if not z.exists():
            drift.append(f"MISSING in zerogen: {name}")
        elif not s.exists():
            drift.append(f"MISSING in NV source: {name}")
        elif sha(z) != sha(s):
            drift.append(f"DRIFT: {name}")

    if drift:
        print("Vendor drift detected:")
        for d in drift:
            print("  " + d)
        return 1
    print(f"OK — all {len(VENDORED)} vendored files in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
