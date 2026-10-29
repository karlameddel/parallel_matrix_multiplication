#!/usr/bin/env python3
"""Generate the input matrices of Assignment 2.

Format: raw little-endian float64, row-major, no header (same as Assignment 1).

DENSE inputs (Part A) are IDENTICAL to Assignment 1: A_n<n>.bin / B_n<n>.bin, uniform in
[0, 1), numpy PCG64 seeded with [SEED, n, tag] (tag 0 -> A, tag 1 -> B). Their SHA-256 can be
compared with the Assignment 1 manifest (--check-a1 path/to/manifest.json).

SPARSE inputs (Part B) are stored as dense files that contain zeros, so that the dense and the
sparse method multiply exactly the same logical matrices (the Java program converts to CSR itself).
File name: data/sparse/A_n<n>_d<density>.bin (dense ones: data/dense/A_n<n>.bin). Density = fraction of nonzero entries (sparsity = 1 - density).
  * exactly k = round(density * n^2) nonzero positions, chosen uniformly at random WITHOUT
    replacement (no structure: no bands, no clusters);
  * nonzero values uniform in (0, 1];
  * numpy PCG64 seeded with [SEED, n, tag, round(density * 1e6)].
For uniformly random inputs the expected density of the product is 1 - (1 - d^2)^n (reported in
the manifest as expected_density_C), so C can be much denser than A and B.
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
SPARSE_DIR = DATA_DIR / "sparse"
CONFIG = ROOT / "config" / "experiment.json"


def dense_matrix(n, tag, seed=SEED):
    return np.random.default_rng([seed, n, tag]).random((n, n), dtype=np.float64)


def sparse_matrix(n, tag, density, seed=SEED):
    rng = np.random.default_rng([seed, n, tag, int(round(density * 1e6))])
    k = int(round(density * n * n))
    m = np.zeros(n * n, dtype=np.float64)
    pos = rng.choice(n * n, size=k, replace=False)
    m[pos] = 1.0 - rng.random(k)          # (0, 1]
    return m.reshape(n, n)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dens_tag(d):
    return f"{d:g}"


def read_json(path, default):
    """JSON file content, or `default` if the file is missing or blank."""
    if path.exists() and path.read_text().strip():
        return json.loads(path.read_text())
    return default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense-sizes", type=int, nargs="*", default=None)
    ap.add_argument("--sparse", nargs="*", default=None, metavar="N:DENSITY",
                    help="e.g. 2000:0.01 (default: all combinations of config/experiment.json)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check-a1", default=None, help="Assignment 1 manifest.json to compare dense hashes with")
    args = ap.parse_args()

    cfg = read_json(CONFIG, {})
    dense_sizes = args.dense_sizes if args.dense_sizes is not None else sorted(
        set(cfg.get("dense", {}).get("sizes", [])) | {cfg.get("dense", {}).get("block_sweep", {}).get("n", 0)} - {0})
    if args.sparse is not None:
        sparse_cfgs = [(int(s.split(":")[0]), float(s.split(":")[1])) for s in args.sparse]
    else:
        sp = cfg.get("sparse", {})
        ds, ss = sp.get("density_sweep", {}), sp.get("size_sweep", {})
        sparse_cfgs = sorted({(ds["n"], d) for d in ds.get("densities", [])} |
                             {(n, ss["density"]) for n in ss.get("sizes", [])}) if ds and ss else []

    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)
    mpath = DATA_DIR / "manifest.json"
    manifest = read_json(mpath, {})
    manifest["_meta"] = {
        "seed": args.seed, "dtype": "float64 little-endian", "layout": "row-major, no header",
        "dense_distribution": "uniform [0, 1)", "dense_seed_list": "[seed, n, tag]",
        "sparse_distribution": "k=round(d*n^2) uniformly random positions without replacement, values uniform (0, 1]",
        "sparse_seed_list": "[seed, n, tag, round(density*1e6)]",
        "generator": "numpy.random.default_rng (PCG64)", "numpy_version": np.__version__,
    }

    for n in dense_sizes:
        for name, tag in (("A", 0), ("B", 1)):
            path = DENSE_DIR / f"{name}_n{n}.bin"
            if path.exists() and not args.force:
                print(f"[skip] {path.name}")
            else:
                dense_matrix(n, tag, args.seed).astype("<f8").tofile(path)
                print(f"[ok]   {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
            manifest.setdefault("dense", {}).setdefault(str(n), {})[name] = sha256(path)

    for n, d in sparse_cfgs:
        key = f"{n}_d{dens_tag(d)}"
        for name, tag in (("A", 0), ("B", 1)):
            path = SPARSE_DIR / f"{name}_n{n}_d{dens_tag(d)}.bin"
            if path.exists() and not args.force:
                print(f"[skip] {path.name}")
            else:
                sparse_matrix(n, tag, d, args.seed).astype("<f8").tofile(path)
                print(f"[ok]   {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
            arr = np.fromfile(path, dtype="<f8")
            manifest.setdefault("sparse", {}).setdefault(key, {})[name] = {
                "sha256": sha256(path), "nnz": int(np.count_nonzero(arr))}
        manifest["sparse"][key]["density"] = d
        manifest["sparse"][key]["expected_density_C"] = 1 - (1 - d * d) ** n

    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Manifest written to {mpath}")

    if args.check_a1:
        a1 = json.loads(Path(args.check_a1).read_text())
        bad = 0
        for n in dense_sizes:
            for name in ("A", "B"):
                ref = a1.get(str(n), {}).get(name)
                got = manifest["dense"][str(n)][name]
                if ref is None:
                    print(f"[a1] n={n} {name}: not in Assignment 1 manifest")
                elif ref != got:
                    bad += 1
                    print(f"[a1] MISMATCH n={n} {name}")
        print("Dense inputs identical to Assignment 1" if bad == 0 else f"{bad} hash mismatches (different NumPy version?)")


if __name__ == "__main__":
    main()