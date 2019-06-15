import neuralNetwork.NeuralNetwork
import neuralNetwork.TestDataUtil

import java.util.Arrays

object MainExecute {

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val inputsFromFile = TestDataUtil.readInputsFromFile("src/main/resources/inputs.txt")
        val outputsFromFile = TestDataUtil.readOutputsFromFile("src/main/resources/outputs.txt")

        val neurons = 5
        val iterations = 50000

        val neuralNetwork = NeuralNetwork(inputsFromFile, outputsFromFile)
        neuralNetwork.addHiddenLayer(neurons)
        neuralNetwork.train(iterations)

        inputsFromFile.forEach { input -> println("Predicted result for [${input[0]},${input[1]}]:${neuralNetwork.predict(arrayOf(input[0], input[1]))} ") }
    }

}

