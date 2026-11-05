# Parallel Matrix Multiplication in Java

**Big Data · Assignment 3 · Academic year 2026-2027**

## Research question

> How does the number of threads affect the runtime, speedup and efficiency of a shared-memory
> parallel blocked matrix multiplication in Java, compared with the equivalent sequential blocked
> implementation of Assignment 2, and when does parallel execution stop helping?

Conclusions are limited to the tested implementation, inputs and hardware. The optional
vectorisation extension was not undertaken.

## Reuse from Assignment 2

| Reused | Where | Note |
|---|---|---|
| Sequential baseline (blocked, bs = 64) | `src/Blocked.java` | Unchanged kernel of Assignment 2 |
| Matrix I/O and checksum | `src/MatrixIO.java`, `src/Checksum.java` | Unchanged |
| Input matrices | `scripts/generate_inputs.py` | Same seeds `[42, n, tag]` -> same bytes; `--check-a2` compares SHA-256 with the Assignment 2 manifest |
| Harness protocol | `scripts/run_benchmarks.py` | One process per configuration, probe + timing run, limits 300 s / 4 GB, median and IQR |
| Correctness criterion | `scripts/verify_correctness.py` | `rtol = 1e-9`, `atol = 1e-12`, NumPy as reference only |

Measurements of Assignment 2 are **not** reused: the baseline is measured again in this session.

## Layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── config/experiment.json        # sizes, thread counts, block size, limits, JVM settings
├── src/                          # Java, no packages
│   ├── Main.java                 # CLI + timing loop (one configuration per process)
│   ├── Blocked.java              # sequential baseline (Assignment 2)
│   ├── ParallelBlocked.java      # parallel version (fixed thread pool)
│   ├── MatrixIO.java
│   └── Checksum.java
├── scripts/
│   ├── generate_inputs.py
│   ├── verify_correctness.py     # run BEFORE benchmarking
│   ├── run_benchmarks.py
│   └── make_report_assets.py     # tables and figures from the raw data
├── data/                         # manifest.json (the .bin inputs are not versioned)
└── results/                      # raw data, tables and figures
```

## Environment

| Component | Value |
|---|---|
| CPU | Intel Core Ultra 9 185H (hybrid: performance, efficient and low-power efficient cores) |
| Physical cores / logical processors | 16 / 22 |
| RAM | 31.7 GB |
| OS | Windows 11 Home 10.0.26200 |
| Java | OpenJDK 21.0.6 (Temurin-21.0.6+7), default JIT, G1 GC, `-Xmx4g` |
| Python | 3.11.9 (only for inputs, harness and analysis; never timed) |
| numpy / pandas / matplotlib / psutil | 2.2.6 / 3.0.0 / 3.10.6 / 7.0.0 |
| Power state | Plugged in, default Windows power mode (same for the whole session) |

## Thread configuration

- `seq`: `Blocked` (Assignment 2, bs = 64), runs on the main thread.
- `par`: `ParallelBlocked`, a fixed pool of exactly p worker threads (`Executors.newFixedThreadPool(p)`),
  one task per strip of 64 rows of C (`ceil(n/64)` tasks: 8, 16, 32 and 63 for n = 500 ... 4000).
  The main thread only submits the tasks and waits.
- Thread counts: 1, 2, 4, 8, 12, 16, 22 and 44. 44 > 22 logical processors, so it is **oversubscribed**
  (marked in the CSV files).
- No CPU affinity is set: the Windows scheduler places the threads on any core.
- No numerical library is used, so there are no hidden library threads. The JVM's own JIT and GC
  threads exist but do not compute the product. `active_java_threads` in `run_status.csv` is p + 1.
- Shared data: each task writes only its own rows of C and only reads A and B, so no locks are needed.
  Every result is bit-identical to `seq` (checked in `verify_correctness.py`).

## Requirements

Windows 11 (the harness reads the Windows peak working set; on Linux it falls back to RSS),
JDK 21 (`javac` and `java` on the PATH), Python 3.11 with `requirements.txt`.
Plug the laptop in, close other applications and do not use the machine during the benchmark.

## How to reproduce (PowerShell, repository root)

```powershell
# 0. Dependencies
python -m pip install -r requirements.txt

# 1. Build
mkdir build
javac -d build (Get-ChildItem src -Filter *.java).FullName

# 2. Inputs (about 330 MB) and check against Assignment 2
python scripts/generate_inputs.py --check-a2 path\to\assignment2\data\manifest.json

# 3. Correctness (before any measurement; must end with ALL TESTS PASSED)
python scripts/verify_correctness.py

# 4. Benchmarks (resumable; --dry-run lists the 36 configurations)
python scripts/run_benchmarks.py --dry-run
python scripts/run_benchmarks.py

# 5. Tables and figures
python scripts/make_report_assets.py
```

Run one configuration by hand:

```powershell
java -Xmx4g -cp build Main --method seq --a data\dense\A_n1000.bin --b data\dense\B_n1000.bin --n 1000 --reps 5 --warmup 3
java -Xmx4g -cp build Main --method par --threads 8 --a data\dense\A_n1000.bin --b data\dense\B_n1000.bin --n 1000 --reps 5 --warmup 3
```

Output: `rep,batch,seconds_per_op,checksum`, then `#key=value` metadata.

## Experimental design in brief

| Aspect | Setting |
|---|---|
| Methods | `seq` (sequential blocked, baseline) and `par` (same blocked kernel, row strips in parallel) |
| Same algorithm | Same tiles (bs = 64), loop order, float64, row-major storage and input files |
| Sizes | n = 500, 1000, 2000, 4000 |
| Threads | p = 1, 2, 4, 8, 12, 16, 22, 44 (44 oversubscribed) |
| Inputs | Assignment 1/2 files, uniform [0, 1), seed `[42, n, tag]` |
| Timed region | Kernel only (zeroing of C included). Loading, allocation of C, creation and shutdown of the thread pool and the checksum are outside |
| Warm-up / batching | ≥ 3 repetitions and ≥ 300 ms; short kernels batched to ≈ 20 ms, reported per multiplication |
| Repetitions | 5 (3 and no warm-up if the probe exceeds 30 s) |
| Summary statistic | Median; variability as IQR |
| Metrics | S(p) = T_seq / T_p, E(p) = S(p) / p, overhead = T_1 / T_seq (all from medians) |
| Memory | Peak working set of the process (psutil, probe run) + live heap after GC + array estimate 24n² bytes |
| Budgets | 300 s and 4 GB per run; violations recorded as results |

Results, analysis and limitations are in the PDF report.

## Output files

`results/raw_measurements.csv`: one row per measured sample
(`timestamp, method, threads, block, n, numeric_type, timer, rep, batch, seconds_per_op, checksum,
warmup_iterations, reps_planned, oversubscribed, affinity`).

`results/run_status.csv`: one row per configuration with status, probe time, peak memory, live heap,
tasks, Java threads, logical processors, physical cores, oversubscription, limits and power state.

`results/correctness.csv`: the correctness tests.

`results/repeat1/`, `results/repeat2/`: two extra runs of `seq` and `par` with 1 and 2 threads
(n = 500, 1000, 2000) in new JVM processes, the second one with `par` before `seq`, to check whether
S(1) ≠ 1 and the superlinear S(2) of the main run are reproducible. Same harness and settings:

```powershell
python scripts/run_benchmarks.py --out-dir results/repeat1 --sizes 500 1000 2000 --threads 1 2
python scripts/run_benchmarks.py --out-dir results/repeat2 --sizes 500 1000 2000 --threads 1 2 --methods par
python scripts/run_benchmarks.py --out-dir results/repeat2 --sizes 500 1000 2000 --methods seq
```

`results/summary_time.csv`, `speedup.csv`, `repeat_check.csv`, `memory.csv`, `limits.csv` and the figures
`time_vs_threads.png`, `speedup.png`, `efficiency.png`: generated by `make_report_assets.py`.

## Use of generative AI

Generative AI (Claude) was used for planning, drafting and debugging code and documentation, in line
with the course policy. See the report for details.