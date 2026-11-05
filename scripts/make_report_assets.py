#!/usr/bin/env python3
"""Tables and figures for Assignment 3 from results/raw_measurements.csv and results/run_status.csv.

Summary statistic: the MEDIAN of the measured samples of each configuration (seconds per
multiplication), with the interquartile range (IQR) as variability.
  T_seq(n) = median time of seq (Blocked, 1 thread)
  T_p(n)   = median time of par with p threads
  S(p) = T_seq(n) / T_p(n)      E(p) = S(p) / p
  overhead of the parallel version with one thread: T_1(n) / T_seq(n)

Usage: python scripts/make_report_assets.py [--in-dir results]
Writes into <in-dir>: summary_time.csv, speedup.csv, memory.csv, limits.csv and
time_vs_threads.png, speedup.png, efficiency.png.
If <in-dir>/repeat*/raw_measurements.csv exist (extra runs of seq and par with 1 and 2 threads in
new JVM processes), also repeat_check.csv: T_seq, T_1, T_2, S(1) and S(2) of every run side by side.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OK = ["ok", "rep_over_time_limit"]


def summarise(df):
    g = df.groupby(["method", "threads", "n"])["seconds_per_op"]
    s = g.agg(reps="count", median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75),
              min="min", max="max").reset_index()
    s["iqr"] = s["q3"] - s["q1"]
    s["iqr_pct_of_median"] = 100 * s["iqr"] / s["median"]
    s["batch"] = df.groupby(["method", "threads", "n"])["batch"].first().values
    s["oversubscribed"] = df.groupby(["method", "threads", "n"])["oversubscribed"].first().values
    return s


def style(ax, xlabel, ylabel, title, threads):
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=str(ROOT / "results"))
    args = ap.parse_args()
    rd = Path(args.in_dir)

    df = pd.read_csv(rd / "raw_measurements.csv")
    st = pd.read_csv(rd / "run_status.csv")
    S = summarise(df)
    S.to_csv(rd / "summary_time.csv", index=False)

    # ------------------------------------------------------------------ speedup and efficiency
    seq = S[S.method == "seq"].set_index("n")
    P = S[S.method == "par"].copy()
    P["t_seq_median"] = P.n.map(seq["median"])
    P["speedup"] = P["t_seq_median"] / P["median"]
    P["efficiency"] = P["speedup"] / P["threads"]
    tasks = st.set_index(["method", "threads", "n"])["tasks_per_multiplication"]
    P["tasks_per_multiplication"] = [tasks.get(("par", t, n)) for t, n in zip(P.threads, P.n)]
    P = P.rename(columns={"median": "t_par_median", "iqr_pct_of_median": "iqr_pct_par"})
    P[["n", "threads", "oversubscribed", "tasks_per_multiplication", "reps", "t_seq_median", "t_par_median",
       "iqr_pct_par", "speedup", "efficiency"]].sort_values(["n", "threads"]).to_csv(rd / "speedup.csv", index=False)

    threads = sorted(P.threads.unique())
    sizes = sorted(S.n.unique())

    # time vs threads (seq drawn as a horizontal line of the same colour)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in sizes:
        g = P[P.n == n].sort_values("threads")
        line = ax.errorbar(g.threads, g.t_par_median, yerr=[g.t_par_median - g.q1, g.q3 - g.t_par_median],
                           marker="o", capsize=2, label=f"par, n={n}")
        ax.axhline(seq.loc[n, "median"], ls="--", lw=.8, color=line[0].get_color())
    ax.set_yscale("log")
    style(ax, "Threads p", "Kernel time per multiplication (s)",
          "Runtime vs threads (median, bars = IQR; dashed = seq)", threads)
    fig.tight_layout(); fig.savefig(rd / "time_vs_threads.png", dpi=150); plt.close(fig)

    # speedup
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in sizes:
        g = P[P.n == n].sort_values("threads")
        ax.plot(g.threads, g.speedup, marker="o", label=f"n={n}")
    ax.plot(threads, threads, "k--", lw=.8, label="ideal S(p) = p")
    ax.axvline(22, color="grey", ls=":", lw=1, label="22 logical processors")
    ax.set_yscale("log", base=2)
    style(ax, "Threads p", "Speedup S(p) = T_seq / T_p", "Speedup over the sequential baseline", threads)
    fig.tight_layout(); fig.savefig(rd / "speedup.png", dpi=150); plt.close(fig)

    # efficiency
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in sizes:
        g = P[P.n == n].sort_values("threads")
        ax.plot(g.threads, g.efficiency, marker="o", label=f"n={n}")
    ax.axhline(1, color="k", ls="--", lw=.8, label="ideal E(p) = 1")
    ax.axvline(22, color="grey", ls=":", lw=1, label="22 logical processors")
    style(ax, "Threads p", "Efficiency E(p) = S(p) / p", "Parallel efficiency", threads)
    fig.tight_layout(); fig.savefig(rd / "efficiency.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ memory and limits
    ok = st[st.status.isin(OK)].copy()
    mem = ok[["method", "threads", "n", "peak_memory_bytes", "live_heap_bytes_after_gc", "array_estimate_bytes",
              "active_java_threads"]].copy()
    for c in ("peak_memory_bytes", "live_heap_bytes_after_gc", "array_estimate_bytes"):
        mem[c.replace("_bytes", "_mb")] = mem[c] / 1e6
    mem.drop(columns=[c for c in mem.columns if c.endswith("_bytes")]).to_csv(rd / "memory.csv", index=False)

    lim = []
    for (m, t), g in st.groupby(["method", "threads"]):
        good = g[g.status.isin(OK)]
        bad = g[~g.status.isin(OK + ["skipped_after_failure"])]
        lim.append({"method": m, "threads": t,
                    "largest_tested_ok_n": int(good.n.max()) if len(good) else None,
                    "largest_attempted_n": int(g[g.status != "skipped_after_failure"].n.max()),
                    "first_failure": (f"n={int(bad.n.min())}: {bad.sort_values('n').status.iloc[0]}" if len(bad) else "")})
    pd.DataFrame(lim).to_csv(rd / "limits.csv", index=False)

    # ------------------------------------------------------------------ repeat runs (between-process check)
    runs = [("main", df)] + [(p.parent.name, pd.read_csv(p)) for p in sorted(rd.glob("repeat*/raw_measurements.csv"))]
    if len(runs) > 1:
        rows = []
        for name, d in runs:
            m = d.groupby(["method", "threads", "n"])["seconds_per_op"].median()
            for n in sorted(d[d.method == "seq"].n.unique()):
                if ("par", 1, n) in m and ("par", 2, n) in m:
                    t_seq, t1, t2 = m[("seq", 1, n)], m[("par", 1, n)], m[("par", 2, n)]
                    rows.append({"run": name, "n": n, "t_seq_median": t_seq, "t_par1_median": t1,
                                 "t_par2_median": t2, "speedup_p1": t_seq / t1, "speedup_p2": t_seq / t2,
                                 "efficiency_p2": t_seq / t2 / 2})
        R = pd.DataFrame(rows).sort_values(["n", "run"])
        R.to_csv(rd / "repeat_check.csv", index=False)

    print(f"tables and figures -> {rd}")


if __name__ == "__main__":
    main()