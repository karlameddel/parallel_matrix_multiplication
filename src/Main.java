import java.io.IOException;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Assignment 3 benchmark driver (one configuration per process).
 *
 * Usage:
 *   java -Xmx4g -cp build Main --method seq|par --a A.bin --b B.bin --n N
 *        [--threads 1] [--block 64] [--reps 5] [--warmup 3] [--warmup-min-ms 300]
 *        [--batch 0] [--target-ms 20] [--out C.bin]
 *
 *   seq  Blocked (sequential baseline of Assignment 2, unchanged). Runs on the main thread.
 *   par  ParallelBlocked with a fixed pool of exactly --threads worker threads.
 *
 * Timed region: the multiplication only (both methods zero C inside it). Input loading, the
 * allocation of C, the creation and shutdown of the thread pool and the checksum are outside.
 * The pool is created once, before the warm-up, and reused by every repetition.
 *
 * Warm-up: at least --warmup repetitions AND at least --warmup-min-ms, excluded from the samples.
 * Batching: --batch 0 (auto) sizes each sample to about --target-ms; the reported time is
 * always seconds PER multiplication.
 *
 * stdout: one CSV line per measured sample  rep,batch,seconds_per_op,checksum
 * followed by "#key=value" metadata lines.
 */
public final class Main {
    private static double[] A, B, C;

    public static void main(String[] args) throws IOException {
        Map<String, String> o = parse(args);
        String method = req(o, "method");
        int n = Integer.parseInt(req(o, "n"));
        int block = Integer.parseInt(o.getOrDefault("block", "64"));
        int threads = Integer.parseInt(o.getOrDefault("threads", "1"));
        int reps = Integer.parseInt(o.getOrDefault("reps", "5"));
        int warmup = Integer.parseInt(o.getOrDefault("warmup", "3"));
        long warmupMinMs = Long.parseLong(o.getOrDefault("warmup-min-ms", "300"));
        int batchOpt = Integer.parseInt(o.getOrDefault("batch", "0"));
        double targetMs = Double.parseDouble(o.getOrDefault("target-ms", "20"));
        if (!method.equals("seq") && !method.equals("par"))
            throw new IllegalArgumentException("unknown method: " + method);
        if (method.equals("seq") && threads != 1)
            throw new IllegalArgumentException("seq always uses 1 thread");
        if (block < 1 || threads < 1) throw new IllegalArgumentException("block and threads must be >= 1");

        A = MatrixIO.load(req(o, "a"), n);
        B = MatrixIO.load(req(o, "b"), n);
        C = new double[n * n];

        // ---- thread pool: created outside the timed region, exactly `threads` workers ----
        ExecutorService pool = method.equals("par") ? Executors.newFixedThreadPool(threads) : null;
        Runnable op = method.equals("seq")
                ? () -> Blocked.multiply(A, B, C, n, block)
                : () -> ParallelBlocked.multiply(A, B, C, n, block, pool);

        // ---- warm-up (excluded) ----
        long wStart = System.nanoTime();
        int w = 0;
        double lastNs = 0;
        while (w < 100000 && (w < warmup || (System.nanoTime() - wStart) / 1e6 < warmupMinMs)) {
            long t0 = System.nanoTime();
            op.run();
            lastNs = System.nanoTime() - t0;
            w++;
        }

        // ---- batch size ----
        int batch;
        if (batchOpt > 0) batch = batchOpt;
        else if (w == 0) batch = 1;                      // no warm-up: no estimate
        else batch = (int) Math.max(1, Math.min(100000, Math.ceil(targetMs * 1e6 / Math.max(1.0, lastNs))));

        // ---- measured samples ----
        for (int r = 0; r < reps; r++) {
            long t0 = System.nanoTime();
            for (int b = 0; b < batch; b++) op.run();
            long t1 = System.nanoTime();
            double secPerOp = (t1 - t0) / 1e9 / batch;
            double checksum = Checksum.sum(C);           // outside the timer, consumes the result
            System.out.printf(Locale.ROOT, "%d,%d,%.9e,%.12e%n", r, batch, secPerOp, checksum);
            System.out.flush();
        }

        int jvmThreads = Thread.activeCount();           // before shutdown: includes the pool
        if (pool != null) pool.shutdown();

        // ---- metadata ----
        System.gc();
        Runtime rt = Runtime.getRuntime();
        long live = rt.totalMemory() - rt.freeMemory();
        System.out.println("#threads=" + (pool == null ? 1 : threads));
        System.out.println("#tasks_per_multiplication=" + (pool == null ? 1 : (n + block - 1) / block));
        System.out.println("#active_java_threads=" + jvmThreads);
        System.out.println("#live_heap_bytes_after_gc=" + live);
        System.out.println("#array_estimate_bytes=" + 24L * n * n);
        System.out.println("#warmup_iterations=" + w);
        System.out.println("#java_version=" + System.getProperty("java.version") + " " + System.getProperty("java.vm.name"));
        System.out.println("#available_processors=" + rt.availableProcessors());
        System.out.println("#max_heap_bytes=" + rt.maxMemory());

        if (o.containsKey("out")) MatrixIO.save(o.get("out"), C, n);
        java.lang.ref.Reference.reachabilityFence(A);
        java.lang.ref.Reference.reachabilityFence(B);
        java.lang.ref.Reference.reachabilityFence(C);
    }

    private static String req(Map<String, String> o, String k) {
        String v = o.get(k);
        if (v == null) throw new IllegalArgumentException("missing --" + k);
        return v;
    }

    private static Map<String, String> parse(String[] args) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i < args.length; i += 2) {
            if (!args[i].startsWith("--") || i + 1 >= args.length)
                throw new IllegalArgumentException("bad argument near '" + args[i] + "'");
            m.put(args[i].substring(2), args[i + 1]);
        }
        return m;
    }
}