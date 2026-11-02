# Parallel Matrix Multiplication in Java

**Big Data · Assignment 3 · Academic year 2026-2027**

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

- Sequential baseline `seq`: `Blocked` (Assignment 2, bs = 64), runs on the main thread.
- Parallel `par`: `ParallelBlocked`, fixed pool of exactly p worker threads (`Executors.newFixedThreadPool(p)`), one task per strip of 64 rows.
- Thread counts: 1, 2, 4, 8, 12, 16, 22, and 44 (44 > 22 logical processors = oversubscription).
- No CPU affinity is set; the Windows scheduler places the threads on any core.
- No numerical library is used, so there are no internal library threads. The JVM's own JIT and GC threads exist but do not compute the product.