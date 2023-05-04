package neuralNetwork

import java.io.IOException
import java.nio.charset.StandardCharsets.*
import java.nio.file.Files
import java.nio.file.Paths

object TestDataUtil {

    @Throws(IOException::class)
    fun readInputsFromFile(file: String): Array<Array<Float>> {
        val lines = Files.readAllLines(Paths.get(file), UTF_8)
        return Array(lines.size) { i ->
            lines[i].split(",".toRegex()).dropLastWhile { it.isEmpty() }.map { it.toFloat() }.toTypedArray()
        }
    }

    @Throws(IOException::class)
    fun readOutputsFromFile(file: String): Array<Int> {
        val lines = Files.readAllLines(Paths.get(file), UTF_8)
        return Array(lines.size) { i -> Integer.parseInt(lines[i]) }
    }
}