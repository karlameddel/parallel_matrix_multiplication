public final class Checksum {
    private Checksum() {}

    public static double sum(double[] v) {
        double s = 0.0;
        for (double x : v) s += x;
        return s;
    }

    public static long countNonZero(double[] v) {
        long c = 0;
        for (double x : v) if (x != 0.0) c++;
        return c;
    }
}
