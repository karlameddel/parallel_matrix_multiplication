import java.util.Arrays;

/**
 * Cache blocking: loop order ikj applied inside bs x bs x bs tiles so that the tiles of B and C
 * stay in cache. bs is a parameter (it may not divide n, or may exceed it).
 * C is zeroed inside the kernel. Per C[i][j] the additions are still in ascending k.
 */
public final class Blocked {
    private Blocked() {}

    public static void multiply(double[] A, double[] B, double[] C, int n, int bs) {
        Arrays.fill(C, 0.0);
        for (int ii = 0; ii < n; ii += bs) {
            int iMax = Math.min(ii + bs, n);
            for (int kk = 0; kk < n; kk += bs) {
                int kMax = Math.min(kk + bs, n);
                for (int jj = 0; jj < n; jj += bs) {
                    int jMax = Math.min(jj + bs, n);
                    for (int i = ii; i < iMax; i++) {
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
}
