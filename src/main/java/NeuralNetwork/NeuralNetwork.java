package NeuralNetwork;

import NeuralNetwork.exceptions.NoHiddenLayersException;

import java.util.ArrayList;
import java.util.stream.IntStream;

public class NeuralNetwork {

    private Float[][] inputs;
    private Integer[] outputs;

    private int numOfInputs;

    private float calcForwardOutput;
    private ResultParser resultParser;
    private Propagator propagator;
    private ArrayList<HiddenLayer> hiddenLayers;

    public NeuralNetwork(Float[][] inputs, Integer[] outputs){
        this.inputs = inputs;
        this.outputs = outputs;
        this.resultParser = new ResultParser();
        this.numOfInputs = inputs[0].length;
        hiddenLayers = new ArrayList<>();
    }

    public void addHiddenLayer(int neurons){
        HiddenLayer hiddenLayer = new HiddenLayer(neurons, this.numOfInputs);
        hiddenLayers.add(hiddenLayer);
    }

    /**
     * Creates a single hidden layer then executes forward and backward propagation
     * @param iterations   Number of times iterated through the network
     * @throws Exception
     */
    public void train(int iterations) throws Exception {
        if (inputs.length != outputs.length)
            throw new Exception(String.format("Inputs and Outputs are not identical. Input Count: %d, Output Count: %d", inputs.length, outputs.length));
        if(hiddenLayers.size() == 0)
            throw new NoHiddenLayersException("Need to add hidden layers!");

        System.out.println("Start process...");
        System.out.println(String.format("Number of Iterations: %d", iterations));
        resultParser.resetNSuccess();
        IntStream.range(0, iterations).forEach(i ->
                IntStream.range(0, outputs.length).forEach(j -> {
                    propagator = new Propagator(hiddenLayers.get(0).getInputWeights(), hiddenLayers.get(0).getOutputWeights(), hiddenLayers.get(0).getNeurons());
                    calcForwardOutput = propagator.forward(inputs[j].clone());
                    propagator.backward(outputs[j], hiddenLayers.get(0).getInputWeights(), hiddenLayers.get(0).getOutputWeights(), hiddenLayers.get(0).getNeurons(), inputs[j].clone());
                    hiddenLayers.get(0).setInputWeights(propagator.getInputWeights());
                    hiddenLayers.get(0).setOutputWeights(propagator.getOutputWeights());
                    resultParser.countSuccess(calcForwardOutput, outputs[j]);
                }));
        final float accuracy = (resultParser.getNSuccess() / (float)(iterations * outputs.length)) * 100;
        System.out.println(String.format("Accuracy: %s", accuracy));
    }

    public int predict(Float[] input){
        return resultParser.parseResult(propagator.forward(input));
    }

}
