package neuralNetwork

import neuralNetwork.exceptions.NoHiddenLayersException
import java.util.concurrent.ThreadLocalRandom
import java.util.stream.IntStream

/**
 * Creates the weights from inputs-to-hidden and hidden-to-outputs
 */
class HiddenLayer(val neurons: Int, inputs: Int) {
    var outputWeights = Array(neurons) { ThreadLocalRandom.current().nextFloat() }
    var inputWeights = Array(inputs) { FloatArray(neurons) { ThreadLocalRandom.current().nextFloat() } }
}

class NeuralNetwork(private val inputs: Array<Array<Float>>, private val outputs: Array<Int>) {

    private val resultHandler: ResultHandler = ResultHandler()
    private val hiddenLayers: ArrayList<HiddenLayer> = ArrayList()
    private var propagator: Propagator? = null

    fun addHiddenLayer(neurons: Int) {
        val hiddenLayer = HiddenLayer(neurons, inputs[0].size)
        hiddenLayers.add(hiddenLayer)
    }

    /**
     * Creates a single hidden layer then executes forward and backward propagation
     * @param iterations   Number of times iterated through the network
     * @throws Exception
     */
    fun train(iterations: Int) {
        if (inputs.size != outputs.size)
            throw Exception("Inputs and Outputs are not identical. Input Count: ${inputs.size}, Output Count: ${outputs.size}")
        if (hiddenLayers.size == 0)
            throw NoHiddenLayersException("Need to add hidden layers!")

        println("Start process...")
        println("Number of Iterations: $iterations")
        resultHandler.resetNSuccess()
        for (i in 0 until iterations) {
            IntStream.range(0, outputs.size).forEach { j ->
                val tempPropagator =
                    Propagator(hiddenLayers[0].inputWeights, hiddenLayers[0].outputWeights, hiddenLayers[0].neurons)
                val calcForwardOutput = tempPropagator.forward(inputs[j].clone())
                tempPropagator.backward(
                    outputs[j].toFloat(),
                    hiddenLayers[0].inputWeights,
                    hiddenLayers[0].outputWeights,
                    hiddenLayers[0].neurons,
                    inputs[j].clone()
                )
                hiddenLayers[0].inputWeights = tempPropagator.inputWeights
                hiddenLayers[0].outputWeights = tempPropagator.outputWeights
                resultHandler.countSuccess(calcForwardOutput, outputs[j].toFloat())
                propagator = tempPropagator
            }
        }
        val accuracy = resultHandler.nSuccess / (iterations * outputs.size).toFloat() * 100
        println("Accuracy: $accuracy")
    }

    fun predict(input: Array<Float>): Int {
        return resultHandler.parseResult(propagator!!.forward(input))
    }

}
