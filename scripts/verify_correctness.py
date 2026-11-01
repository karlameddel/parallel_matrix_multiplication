#!/usr/bin/env python3
"""Correctness validation for Assignment 3 (run BEFORE benchmarking).

Both methods are checked against NumPy (A @ B), which is used ONLY as a reference:
  seq  Blocked (sequential baseline of Assignment 2)
  par  ParallelBlocked with several thread counts, including 1 and more threads than tasks
       or than logical processors (oversubscription)
Cases: a hand-computed 2x2 product; A*I = A, I*B = B, A*0 = 0; random dense matrices (same
generator and seeds as the benchmark inputs); n = 1 and sizes that the block size does not divide.
Block sizes 7, 64 and 300 (does not divide n / default / larger than n).

Pass criterion (same as Assignments 1 and 2): |x - ref| <= ATOL + RTOL*|ref| for every entry,
RTOL = 1e-9, ATOL = 1e-12. All entries are >= 0, so there is no cancellation and the rounding error
of a sum of n terms (about n*2^-53 relative) is far below the tolerance.

Safe access to shared data: every par result must also be BIT-IDENTICAL to the seq result with the
same block size (each C[i][j] receives the same additions in the same order), and par runs 3
repetitions in the same process whose checksums must be identical. A data race would break this.

Usage: python scripts/verify_correctness.py [--sizes 1 10 100 200] [--threads 1 2 3 8 22 44]
Writes results/correctness.csv; exit code 1 on any failure.
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_inputs import dense_matrix  # noqa: E402

BUILD = ROOT / "build"
OUT_CSV = ROOT / "results" / "correctness.csv"
RTOL, ATOL = 1e-9, 1e-12
BLOCKS = [7, 64, 300]


def run(method, threads, a, b, n, out, block):
    reps = "3" if method == "par" else "1"
    cmd = ["java", "-cp", str(BUILD), "Main", "--method", method, "--threads", str(threads),
           "--a", str(a), "--b", str(b), "--n", str(n), "--block", str(block), "--reps", reps,
           "--warmup", "0", "--warmup-min-ms", "0", "--batch", "1", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])
    sums = [float(l.split(",")[3]) for l in r.stdout.splitlines() if l and not l.startswith("#")]
    c = np.fromfile(out, dtype="<f8")
    if c.size != n * n:
        raise RuntimeError(f"output has {c.size} values, expected {n * n}")
    return c.reshape(n, n), sums


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 100, 200])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 3, 8, 22, 44])
    args = ap.parse_args()
    if not (BUILD / "Main.class").exists():
        sys.exit("missing build/Main.class: compile first (see README)")

    cases = [("known_2x2", 2, np.array([[1., 2.], [3., 4.]]), np.array([[5., 6.], [7., 8.]]),
              np.array([[19., 22.], [43., 50.]]))]                       # reference computed by hand
    for n in args.sizes:
        A, B = dense_matrix(n, 0), dense_matrix(n, 1)
        I, Z = np.eye(n), np.zeros((n, n))
        cases += [("A_times_identity", n, A, I, A), ("identity_times_B", n, I, B, B),
                  ("A_times_zero", n, A, Z, Z), ("random_dense", n, A, B, A @ B)]

    rows, all_ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a_p, b_p, c_p = tmp / "A.bin", tmp / "B.bin", tmp / "C.bin"
        for case, n, A, B, ref in cases:
            A.astype("<f8").tofile(a_p)
            B.astype("<f8").tofile(b_p)
            for bs in BLOCKS:
                base = None
                for method, p in [("seq", 1)] + [("par", t) for t in args.threads]:
                    try:
                        c, sums = run(method, p, a_p, b_p, n, c_p, bs)
                        err = np.abs(c - ref)
                        nz = ref != 0
                        abs_e = float(err.max())
                        rel = float((err[nz] / np.abs(ref[nz])).max()) if nz.any() else 0.0
                        ok = bool(np.all(err <= ATOL + RTOL * np.abs(ref)))
                        if method == "seq":
                            base = c
                            same = ""
                        else:
                            same = bool(np.array_equal(c, base))
                            ok &= same and len(set(sums)) == 1
                    except Exception as e:
                        abs_e = rel = float("nan"); ok = False; same = ""
                        print(f"  ERROR {method}/{p}/{case}/n={n}: {e}")
                    all_ok &= ok
                    rows.append([case, n, method, p, bs, f"{abs_e:.3e}", f"{rel:.3e}", same,
                                 RTOL, ATOL, "PASS" if ok else "FAIL"])
                    print(f"[{'PASS' if ok else 'FAIL'}] {case:17s} n={n:<4d} bs={bs:<4d} {method} p={p:<3d}"
                          f" max_abs={abs_e:.2e} max_rel={rel:.2e} bit_identical_to_seq={same if same != '' else '-'}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "n", "method", "threads", "block", "max_abs_error", "max_rel_error",
                    "bit_identical_to_seq", "rtol", "atol", "result"])
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'} ({len(rows)} tests)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()