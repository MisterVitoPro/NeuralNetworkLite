package NeuralNetwork;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class TestDataUtil {

    public static Float[][] readInputsFromFile(String file) throws IOException {
        final List<String> lines = Files.readAllLines(Paths.get(file), StandardCharsets.UTF_8);
        final Float[][] fArray = new Float[lines.size()][];
        IntStream.range(0, lines.size()).forEach(i -> fArray[i] =  Arrays.stream(lines.get(i).split(",")).map(Float::valueOf).toArray(Float[]::new));
        return fArray;
    }

    public static Integer[] readOutputsFromFile(String file) throws IOException {
        final List<String> lines = Files.readAllLines(Paths.get(file), StandardCharsets.UTF_8);
        final Integer[] iArray = new Integer[lines.size()];
        IntStream.range(0, lines.size()).forEach(i -> iArray[i] = Integer.parseInt(lines.get(i)));
        return iArray;
    }
}
