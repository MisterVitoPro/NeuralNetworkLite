package neuralNetwork.activationfunction

import kotlin.math.exp

class Sigmoid : ActivationFunction {

    override fun activate(x: Float): Float {
        return (1.0f / (1.0f + exp(-x)))
    }

    override fun derivative(x: Float): Float {
        return activate(x) * (1 - activate(x))
    }
}