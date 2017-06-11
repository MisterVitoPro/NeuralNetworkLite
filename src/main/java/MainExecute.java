import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.TestDataUtil;

public class MainExecute {

    public static void main(String [] args) throws Exception {
        Float[][] inputsFromFile = TestDataUtil.readInputsFromFile("src/main/resources/inputs.txt");
        Integer[] outputsFromFile = TestDataUtil.readOutputsFromFile("src/main/resources/outputs.txt");

        int neurons = 10;
        int iterations = 10000;

        NeuralNetwork neuralNetwork = new NeuralNetwork(inputsFromFile, outputsFromFile, neurons);
        neuralNetwork.train(iterations);

        System.out.println("Predicted result for [1,1]: " + neuralNetwork.predict(new Float[] {1f, 1f}));
        System.out.println("Predicted result for [1,0]: " + neuralNetwork.predict(new Float[] {1f, 0f}));
        System.out.println("Predicted result for [0,1]: " + neuralNetwork.predict(new Float[] {0f, 1f}));
        System.out.println("Predicted result for [0,0]: " + neuralNetwork.predict(new Float[] {0f, 0f}));
    }

}

