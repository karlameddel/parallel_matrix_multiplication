import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

/** Reads and writes dense n x n float64 matrices (little-endian, row-major, no header). */
public final class MatrixIO {
    private MatrixIO() {}

    public static double[] load(String path, int n) throws IOException {
        long expected = (long) n * n * 8L;
        if (Files.size(Paths.get(path)) != expected)
            throw new IOException(path + ": size " + Files.size(Paths.get(path)) + " B, expected " + expected + " B for n=" + n);
        double[] m = new double[n * n];
        try (FileChannel ch = FileChannel.open(Paths.get(path), StandardOpenOption.READ)) {
            ByteBuffer buf = ByteBuffer.allocate(n * n * 8).order(ByteOrder.LITTLE_ENDIAN);
            while (buf.hasRemaining()) if (ch.read(buf) < 0) break;
            buf.flip();
            buf.asDoubleBuffer().get(m);
        }
        return m;
    }

    public static void save(String path, double[] m, int n) throws IOException {
        ByteBuffer out = ByteBuffer.allocate(n * n * 8).order(ByteOrder.LITTLE_ENDIAN);
        out.asDoubleBuffer().put(m);
        Files.write(Paths.get(path), out.array());
    }
}
