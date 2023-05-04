import neuralNetwork.NeuralNetwork
import neuralNetwork.TestDataUtil

fun main() {
    val inputsFromFile = TestDataUtil.readInputsFromFile("src/main/resources/inputs.txt")
    val outputsFromFile = TestDataUtil.readOutputsFromFile("src/main/resources/outputs.txt")

    val neurons = 5
    val iterations = 10000

    val neuralNetwork = NeuralNetwork(inputsFromFile, outputsFromFile)
    neuralNetwork.addHiddenLayer(neurons)
    neuralNetwork.train(iterations)

    inputsFromFile.forEach { input ->
        println(
            "Predicted result for [${input[0]},${input[1]}]:${
                neuralNetwork.predict(
                    arrayOf(input[0], input[1])
                )
            } "
        )
    }
}


