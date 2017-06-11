package NeuralNetwork;

import java.util.stream.IntStream;

public class NeuralNetwork {

    private Float[][] inputs;
    private Integer[] outputs;

    private int hiddenNeurons;
    private int numOfInputs;

    private float calcForwardOutput;
    private ResultParser resultParser;
    private Propagator propagator;

    public NeuralNetwork(Float[][] inputs, Integer[] outputs, int hiddenNeurons){
        this.inputs = inputs;
        this.outputs = outputs;
        this.resultParser = new ResultParser();
        this.numOfInputs = inputs[0].length;
        this.hiddenNeurons = hiddenNeurons;
    }

    /**
     * Creates a single hidden layer then executes forward and backward propagation
     * @param iterations   Number of times iterated through the network
     * @throws Exception
     */
    public void train(int iterations) throws Exception {
        if (inputs.length != outputs.length)
            throw new Exception(String.format("Inputs and Outputs are not identical. Input Count: %d, Output Count: %d", inputs.length, outputs.length));

        HiddenLayer hiddenLayer = new HiddenLayer(this.hiddenNeurons, this.numOfInputs);

        System.out.println("Start process...");
        System.out.println(String.format("Number of Iterations: %d", iterations));
        resultParser.resetNSuccess();
        IntStream.range(0, iterations).forEach(i ->
                IntStream.range(0, outputs.length).forEach(j -> {
                    propagator = new Propagator(hiddenLayer.getInputWeights(), hiddenLayer.getOutputWeights(), hiddenNeurons);
                    calcForwardOutput = propagator.forward(inputs[j].clone());
                    propagator.backward(outputs[j], hiddenLayer.getInputWeights(), hiddenLayer.getOutputWeights(), hiddenNeurons, inputs[j].clone());
                    hiddenLayer.setInputWeights(propagator.getInputWeights());
                    hiddenLayer.setOutputWeights(propagator.getOutputWeights());
                    resultParser.countSuccess(calcForwardOutput, outputs[j]);
                }));
        final float accuracy = (resultParser.getNSuccess() / (float)(iterations * outputs.length)) * 100;
        System.out.println(String.format("Accuracy: %s", accuracy));
    }

    public int predict(Float[] input){
        return resultParser.parseResult(propagator.forward(input));
    }

}
