#!/usr/bin/env python3
"""Benchmark harness for Assignment 3 (sequential blocked vs parallel blocked, several thread counts).

Same protocol as Assignments 1 and 2:
  * every configuration runs in its own JVM process;
  * probe run (1 repetition, no warm-up) -> time of one multiplication and peak memory
    (psutil, polled every 10 ms; Windows peak working set summed over the process tree);
  * timing run: warm-up (>= N repetitions AND >= warmup_min_ms, excluded) + measured repetitions,
    all inside one process. Short kernels are batched automatically (time reported PER operation);
  * if the probe takes > costly_s seconds: 3 repetitions and no warm-up (reported);
  * limits: time_s per run and mem_bytes of peak memory. A violation is recorded as a result
    (timeout / memory_exceeded) without any invented time; larger n of the same series is skipped;
  * checksums of all configurations on the same input are compared and mismatches are reported.
Differences from Assignment 2: NO CPU affinity is set (the parallel runs need all logical
processors, and seq is run under the same conditions); configurations are (method, threads, n).
A configuration is marked oversubscribed when threads > logical processors.

Configurations come from config/experiment.json: for every n, seq (1 thread) and par with every
thread count of "threads".

Outputs (append mode, resumable; default results/):
  raw_measurements.csv   one row per measured sample
  run_status.csv         one row per configuration: status, probe time, memory, threads, power state

Usage (repo root):
  python scripts/run_benchmarks.py --dry-run
  python scripts/run_benchmarks.py --out-dir results/test --sizes 500 --threads 1 4
  python scripts/run_benchmarks.py
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
DATA, BUILD, CONFIG = ROOT / "data" / "dense", ROOT / "build", ROOT / "config" / "experiment.json"
LOGICAL = psutil.cpu_count(logical=True)
PHYSICAL = psutil.cpu_count(logical=False)

RAW_HEADER = ["timestamp", "method", "threads", "block", "n", "numeric_type", "timer", "rep", "batch",
              "seconds_per_op", "checksum", "warmup_iterations", "reps_planned", "oversubscribed", "affinity"]
STATUS_HEADER = ["timestamp", "method", "threads", "block", "n", "status", "probe_seconds",
                 "peak_memory_bytes", "memory_metric", "live_heap_bytes_after_gc", "array_estimate_bytes",
                 "reps", "warmup_planned", "warmup_iterations", "tasks_per_multiplication",
                 "active_java_threads", "available_processors", "logical_processors", "physical_cores",
                 "oversubscribed", "time_limit_s", "mem_limit_bytes", "affinity", "power_plugged",
                 "battery_percent", "java_version"]


# ----------------------------------------------------------------------------- configurations
def build_configs(cfg, sizes=None, threads=None, methods=None):
    out = []
    for n in sizes or cfg["sizes"]:
        out.append({"method": "seq", "threads": 1, "n": n})
        for p in threads or cfg["threads"]:
            out.append({"method": "par", "threads": p, "n": n})
    return [c for c in out if not methods or c["method"] in methods]


def key_of(c):
    return f"{c['method']}|{c['threads']}|{c['n']}"


def series(c):
    return (c["method"], c["threads"])


# ----------------------------------------------------------------------------- process handling
def sample_memory(ps):
    total = 0
    try:
        procs = [ps] + ps.children(recursive=True)
    except psutil.Error:
        return 0
    for p in procs:
        try:
            mi = p.memory_info()
            total += getattr(mi, "peak_wset", mi.rss)      # Windows: peak working set; else RSS
        except psutil.Error:
            pass
    return total


def kill_tree(proc):
    try:
        for ch in psutil.Process(proc.pid).children(recursive=True):
            ch.kill()
    except psutil.Error:
        pass
    proc.kill()


def run_monitored(cmd, timeout_s, mem_limit):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ps = psutil.Process(proc.pid)
    start, peak, status = time.perf_counter(), 0, "ok"
    while proc.poll() is None:
        peak = max(peak, sample_memory(ps))
        if peak > mem_limit:
            status = "memory_exceeded"
            break
        if time.perf_counter() - start > timeout_s:
            status = "timeout"
            break
        time.sleep(0.01)
    if status != "ok":
        kill_tree(proc)
    out, err = proc.communicate()
    if status == "ok" and proc.returncode != 0:
        status = "error"
    return status, out, err, peak


def parse(out):
    rows, meta = [], {}
    for line in out.strip().splitlines():
        if line.startswith("#"):
            k, _, v = line[1:].partition("=")
            meta[k] = v
        elif line:
            p = line.split(",")
            rows.append({"rep": int(p[0]), "batch": int(p[1]), "seconds": float(p[2]), "checksum": float(p[3])})
    return rows, meta


def append(path, header, row):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def load_status(path):
    done, blocked = set(), set()
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                c = {"method": r["method"], "threads": int(r["threads"]), "n": int(r["n"])}
                done.add(key_of(c))
                if r["status"] != "ok":
                    blocked.add(series(c))
    return done, blocked


def power_state():
    try:
        b = psutil.sensors_battery()
    except Exception:
        b = None
    if b is None:
        return "unknown", ""
    return str(b.power_plugged), f"{b.percent:.0f}"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--sizes", type=int, nargs="+", help="restrict to these n")
    ap.add_argument("--threads", type=int, nargs="+", help="restrict par to these thread counts")
    ap.add_argument("--methods", nargs="+", help="restrict to these methods (seq par)")
    ap.add_argument("--reps", type=int, help="override measured repetitions")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--dry-run", action="store_true", help="list the configurations and exit")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    lim, jv, block, aff = cfg["limits"], cfg["java"], cfg["block"], cfg.get("affinity", "none")
    reps_cfg = args.reps or jv["reps"]
    time_lim, mem_lim, costly = lim["time_s"], lim["mem_bytes"], lim["costly_s"]

    configs = build_configs(cfg, args.sizes, args.threads, args.methods)
    print(f"Logical processors: {LOGICAL}, physical cores: {PHYSICAL}, affinity: {aff}")
    if args.dry_run:
        for c in configs:
            print(key_of(c), "(oversubscribed)" if c["threads"] > LOGICAL else "")
        print(f"{len(configs)} configurations")
        return
    if not (BUILD / "Main.class").exists():
        sys.exit("missing build/Main.class: compile first (see README)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path, status_path = out_dir / "raw_measurements.csv", out_dir / "run_status.csv"
    done, blocked = load_status(status_path)
    checks = {}

    for c in configs:
        k, n, p = key_of(c), c["n"], c["threads"]
        over = p > LOGICAL
        ts = datetime.now().isoformat(timespec="seconds")
        if k in done:
            print(f"[skip] {k} already done")
            continue
        plugged, batt = power_state()

        def status_row(status, probe_s="", peak="", meta=None, reps="", wu="", wu_it=""):
            meta = meta or {}
            return [ts, c["method"], p, block, n, status, probe_s, peak, "peak_working_set",
                    meta.get("live_heap_bytes_after_gc", ""), meta.get("array_estimate_bytes", ""), reps, wu,
                    wu_it, meta.get("tasks_per_multiplication", ""), meta.get("active_java_threads", ""),
                    meta.get("available_processors", ""), LOGICAL, PHYSICAL, over, time_lim, mem_lim, aff,
                    plugged, batt, meta.get("java_version", "")]

        if series(c) in blocked:
            append(status_path, STATUS_HEADER, status_row("skipped_after_failure"))
            print(f"[skip] {k}: a smaller size of this series already failed")
            continue
        a, b = DATA / f"A_n{n}.bin", DATA / f"B_n{n}.bin"
        if not (a.exists() and b.exists()):
            sys.exit(f"missing input {a.name}/{b.name}: run scripts/generate_inputs.py")

        base = ["java", f"-Xmx{jv['xmx']}", "-cp", str(BUILD), "Main", "--method", c["method"],
                "--threads", str(p), "--a", str(a), "--b", str(b), "--n", str(n), "--block", str(block)]

        print(f"[probe] {k} ...", flush=True)
        st, out, err, peak = run_monitored(base + ["--reps", "1", "--warmup", "0", "--warmup-min-ms", "0",
                                                  "--batch", "1"], time_lim + 60, mem_lim)
        if st == "error":
            sys.exit(f"{k} failed: {err.strip()[:500]}")
        rows, meta = parse(out) if st == "ok" else ([], {})
        probe_s = rows[0]["seconds"] if rows else ""
        if st == "ok" and probe_s > time_lim:
            st = "timeout"
        if st != "ok":
            append(status_path, STATUS_HEADER, status_row(st, probe_s, peak, meta, 0, 0))
            blocked.add(series(c))
            print(f"  -> {st} (peak {peak / 1e6:.0f} MB)")
            continue

        if probe_s > costly:
            reps, warmup, wmin = 3, 0, 0
        else:
            reps, warmup, wmin = reps_cfg, jv["warmup"], jv["warmup_min_ms"]
        print(f"  probe {probe_s:.4f}s, peak {peak / 1e6:.0f} MB -> {reps} reps + >= {warmup} warm-up", flush=True)
        budget = time_lim * (reps + warmup + 1) + 60 + wmin / 1000
        st2, out, err, _ = run_monitored(base + ["--reps", str(reps), "--warmup", str(warmup),
                                                  "--warmup-min-ms", str(wmin), "--target-ms", str(jv["target_ms"])],
                                         budget, mem_lim)
        if st2 != "ok":
            append(status_path, STATUS_HEADER, status_row(st2 + "_timing_run", probe_s, peak, meta, reps, warmup))
            blocked.add(series(c))
            print(f"  -> {st2} in timing run {err.strip()[:200] if st2 == 'error' else ''}")
            continue
        rows, meta2 = parse(out)
        for r in rows:
            append(raw_path, RAW_HEADER, [ts, c["method"], p, block, n, "float64", "kernel_only", r["rep"],
                   r["batch"], f"{r['seconds']:.9e}", f"{r['checksum']:.12e}",
                   meta2.get("warmup_iterations", ""), reps, over, aff])
        if len({r["checksum"] for r in rows}) != 1:
            print(f"  WARNING: checksum changes between repetitions for {k}")
        checks.setdefault(n, {})[k] = rows[0]["checksum"]
        if len(set(checks[n].values())) != 1:
            print(f"  WARNING: checksums differ across configurations for n={n}: {checks[n]}")
        final = "ok" if max(r["seconds"] for r in rows) <= time_lim else "rep_over_time_limit"
        append(status_path, STATUS_HEADER, status_row(final, probe_s, peak, meta2 or meta, reps, warmup,
                                                      meta2.get("warmup_iterations", "")))
        t = sorted(r["seconds"] for r in rows)
        print(f"  median {t[len(t) // 2]:.6g} s (batch {rows[0]['batch']})")


if __name__ == "__main__":
    main()