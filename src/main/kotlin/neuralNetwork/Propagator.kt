package neuralNetwork

import neuralNetwork.activationfunction.ActivationFunction
import neuralNetwork.activationfunction.Sigmoid
import java.util.concurrent.atomic.AtomicInteger
import java.util.stream.IntStream
import kotlin.math.exp

class Propagator(val inputWeights: Array<FloatArray>, outputWeightsParam: Array<Float>, private val numOfNeurons: Int, private val activateFunction: ActivationFunction) {

    private val hiddenOutputActivated: Array<Float> = Array(numOfNeurons) { 0f }
    private val hiddenInputWeightSum: Array<Float> = Array(numOfNeurons) { 0f }
    private var outputSum: Float = 0f
    private var calculatedOutput: Float = 0f
    var outputWeights: Array<Float> = outputWeightsParam

    fun forward(inputs: Array<Float>): Float {

        for (i in 0 until numOfNeurons) {
            val aInteger = AtomicInteger(0)
            hiddenInputWeightSum[i] = inputs.fold(0f) { a, b -> a + b * inputWeights[aInteger.getAndIncrement()][i] }
            hiddenOutputActivated[i] = activateFunction.activate(hiddenInputWeightSum[i])
        }

        val integer = AtomicInteger(0)
        outputSum = hiddenOutputActivated.fold(0f) { a, b -> a + b * outputWeights[integer.getAndIncrement()] }
        calculatedOutput = activateFunction.activate(outputSum)
        return calculatedOutput
    }

    fun backward(
        target: Float,
        inputWeights: Array<FloatArray>,
        outputWeights: Array<Float>,
        neurons: Int,
        inputs: Array<Float>
    ) {
        val marginOfError = target - calculatedOutput
        val deltaOutputSum = activateFunction.derivative(outputSum) * marginOfError

        val atomicInteger = AtomicInteger(0)
        this.outputWeights = outputWeights.map { out ->
            val deltaWeight = deltaOutputSum * hiddenOutputActivated[atomicInteger.getAndIncrement()]
            out + deltaWeight
        }.toTypedArray()

        val deltaHiddenSum = FloatArray(neurons)
        val deltaWeights = Array(inputs.size) { FloatArray(neurons) }
        IntStream.range(0, numOfNeurons).forEach { i ->
            deltaHiddenSum[i] = deltaOutputSum * outputWeights[i] * activateFunction.derivative(hiddenInputWeightSum[i])
            IntStream.range(0, inputs.size).forEach { j ->
                deltaWeights[j][i] = deltaHiddenSum[i] * inputs[j]
                this.inputWeights[j][i] = inputWeights[j][i] + deltaWeights[j][i]
            }
        }
    }
}
