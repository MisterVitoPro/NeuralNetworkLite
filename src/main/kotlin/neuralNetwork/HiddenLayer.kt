package neuralNetwork

import java.util.concurrent.ThreadLocalRandom
import java.util.stream.IntStream

/**
 * Creates the weights from inputs-to-hidden and hidden-to-outputs
 */
class HiddenLayer(val neurons: Int, inputs: Int) {

    var outputWeights = Array(neurons) { ThreadLocalRandom.current().nextFloat() }
    var inputWeights = Array(inputs) { FloatArray(neurons) {ThreadLocalRandom.current().nextFloat()} }

}
