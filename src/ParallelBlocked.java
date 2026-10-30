import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

/**
 * Parallel version of Blocked: same tiles (bs), same i-k-j order inside the tiles, float64, row-major.
 * Each strip of bs rows (one value of the ii loop) is one task. A task writes only its own rows
 * of C and only reads A and B, so no two threads write the same element and no locks are needed.
 * For every C[i][j] the products are still added in ascending k, so the result is expected to be
 * bit-identical to Blocked.
 * The thread pool is created by the caller (outside the timed region) and has exactly p threads.
 */
public final class ParallelBlocked {
    private ParallelBlocked() {}

    public static void multiply(double[] A, double[] B, double[] C, int n, int bs, ExecutorService pool) {
        List<Callable<Void>> tasks = new ArrayList<>();
        for (int ii = 0; ii < n; ii += bs) {
            final int i0 = ii, i1 = Math.min(ii + bs, n);
            tasks.add(() -> { strip(A, B, C, n, bs, i0, i1); return null; });
        }
        try {
            for (Future<Void> f : pool.invokeAll(tasks)) f.get();   // waits for all strips
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    /** Rows i0..i1-1 of C: the same loops as Blocked for one value of ii. */
    private static void strip(double[] A, double[] B, double[] C, int n, int bs, int i0, int i1) {
        Arrays.fill(C, i0 * n, i1 * n, 0.0);
        for (int kk = 0; kk < n; kk += bs) {
            int kMax = Math.min(kk + bs, n);
            for (int jj = 0; jj < n; jj += bs) {
                int jMax = Math.min(jj + bs, n);
                for (int i = i0; i < i1; i++) {
                    int co = i * n;
                    for (int k = kk; k < kMax; k++) {
                        double a = A[i * n + k];
                        int bo = k * n;
                        for (int j = jj; j < jMax; j++)
                            C[co + j] += a * B[bo + j];
                    }
                }
            }
        }
    }
}