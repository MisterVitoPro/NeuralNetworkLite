package neuralNetwork

import neuralNetwork.exceptions.NoHiddenLayersException

import java.util.ArrayList
import java.util.stream.IntStream

class NeuralNetwork(private val inputs: Array<Array<Float>>, private val outputs: Array<Int>) {

    private val numOfInputs: Int = inputs[0].size

    private var calcForwardOutput: Float = 0f
    private val resultParser: ResultParser = ResultParser()
    private var propagator: Propagator? = null
    private val hiddenLayers: ArrayList<HiddenLayer> = ArrayList()

    fun addHiddenLayer(neurons: Int) {
        val hiddenLayer = HiddenLayer(neurons, this.numOfInputs)
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
        resultParser.resetNSuccess()
        for(i in 0 until iterations) {
            IntStream.range(0, outputs.size).forEach { j ->
                propagator = Propagator(hiddenLayers[0].inputWeights, hiddenLayers[0].outputWeights, hiddenLayers[0].neurons)
                calcForwardOutput = propagator!!.forward(inputs[j].clone())
                propagator!!.backward(outputs[j].toFloat(), hiddenLayers[0].inputWeights, hiddenLayers[0].outputWeights, hiddenLayers[0].neurons, inputs[j].clone())
                hiddenLayers[0].inputWeights = propagator!!.inputWeights
                hiddenLayers[0].outputWeights = propagator!!.outputWeights
                resultParser.countSuccess(calcForwardOutput, outputs[j].toFloat())
            }
        }
        val accuracy = resultParser.nSuccess / (iterations * outputs.size).toFloat() * 100
        println("Accuracy: $accuracy")
    }

    fun predict(input: Array<Float>): Int {
        return resultParser.parseResult(propagator!!.forward(input))
    }

}
