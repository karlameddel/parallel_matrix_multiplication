#!/usr/bin/env python3
"""Generate the dense input matrices of Assignment 3.

The files are IDENTICAL to Assignments 1 and 2: data/dense/A_n<n>.bin and B_n<n>.bin,
uniform values in [0, 1), numpy PCG64 seeded with [SEED, n, tag] (tag 0 -> A, tag 1 -> B),
stored as raw little-endian float64, row-major, no header.
The SHA-256 of every file is written to data/manifest.json. With --check-a2 the hashes are
compared with the manifest of Assignment 2 to prove that the inputs are the same.

Usage (repo root):
  python scripts/generate_inputs.py
  python scripts/generate_inputs.py --sizes 500 1000 --check-a2 path/to/assignment2/data/manifest.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DENSE_DIR = DATA_DIR / "dense"
CONFIG = ROOT / "config" / "experiment.json"
DEFAULT_SIZES = [500, 1000, 2000, 4000]


def dense_matrix(n, tag, seed=SEED):
    return np.random.default_rng([seed, n, tag]).random((n, n), dtype=np.float64)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path, default):
    """JSON file content, or `default` if the file is missing or blank."""
    if path.exists() and path.read_text().strip():
        return json.loads(path.read_text())
    return default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="default: 'sizes' of config/experiment.json")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--check-a2", default=None, help="Assignment 2 data/manifest.json to compare hashes with")
    args = ap.parse_args()

    sizes = args.sizes or read_json(CONFIG, {}).get("sizes", DEFAULT_SIZES)
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    mpath = DATA_DIR / "manifest.json"
    manifest = read_json(mpath, {})
    manifest["_meta"] = {
        "seed": args.seed, "dtype": "float64 little-endian", "layout": "row-major, no header",
        "distribution": "uniform [0, 1)", "seed_list": "[seed, n, tag]  (tag 0 = A, 1 = B)",
        "generator": "numpy.random.default_rng (PCG64)", "numpy_version": np.__version__,
    }

    for n in sizes:
        for name, tag in (("A", 0), ("B", 1)):
            path = DENSE_DIR / f"{name}_n{n}.bin"
            if path.exists() and not args.force:
                print(f"[skip] {path.name}")
            else:
                dense_matrix(n, tag, args.seed).astype("<f8").tofile(path)
                print(f"[ok]   {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
            manifest.setdefault("dense", {}).setdefault(str(n), {})[name] = sha256(path)

    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Manifest written to {mpath}")

    if args.check_a2:
        a2 = json.loads(Path(args.check_a2).read_text()).get("dense", {})
        bad = 0
        for n in sizes:
            for name in ("A", "B"):
                ref, got = a2.get(str(n), {}).get(name), manifest["dense"][str(n)][name]
                if ref is None:
                    print(f"[a2] n={n} {name}: not in the Assignment 2 manifest")
                elif ref != got:
                    bad += 1
                    print(f"[a2] MISMATCH n={n} {name}")
        print("Inputs identical to Assignment 2" if bad == 0 else f"{bad} hash mismatches (different NumPy version?)")


if __name__ == "__main__":
    main()