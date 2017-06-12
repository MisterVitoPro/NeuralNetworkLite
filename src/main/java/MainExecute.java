import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.TestDataUtil;

import java.util.Arrays;

public class MainExecute {

    public static void main(String [] args) throws Exception {
        Float[][] inputsFromFile = TestDataUtil.readInputsFromFile("src/main/resources/inputs.txt");
        Integer[] outputsFromFile = TestDataUtil.readOutputsFromFile("src/main/resources/outputs.txt");

        int neurons = 10;
        int iterations = 10000;

        NeuralNetwork neuralNetwork = new NeuralNetwork(inputsFromFile, outputsFromFile);
        neuralNetwork.addHiddenLayer(neurons);
        neuralNetwork.train(iterations);

        Arrays.stream(inputsFromFile).forEach(input ->
                System.out.println(String.format("Predicted result for [%s,%s]:%s ", input[0], input[1], neuralNetwork.predict(new Float[] {input[0], input[1]}))));
    }

}

