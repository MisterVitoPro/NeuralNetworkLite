package neuralNetwork.activationfunction

interface ActivationFunction {

    fun activate(x: Float): Float

    fun derivative(x: Float): Float
}