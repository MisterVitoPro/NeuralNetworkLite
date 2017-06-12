package NeuralNetwork;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class Propagator {

    private HiddenLayer[] hiddenLayers;
    private Float[] hiddenOutputActivated;
    private Float[] hiddenInputWeightSum;
    private float outputSum;
    private float calculatedOutput;
    private float[][] inputWeights;
    private Float[] outputWeights;
    private int numOfNeurons;

    public Propagator(float[][] inputWeights, Float[] outputWeights, int numOfNeurons){
        this.inputWeights = inputWeights;
        this.outputWeights = outputWeights;
        this.numOfNeurons = numOfNeurons;
        hiddenOutputActivated = new Float[numOfNeurons];
        hiddenInputWeightSum = new Float[numOfNeurons];
    }

    public float forward(Float[] inputs){
        IntStream.range(0, numOfNeurons).forEach(i -> {
            AtomicInteger aInteger = new AtomicInteger(0);
            hiddenInputWeightSum[i] = Arrays.stream(inputs).reduce(0f, (a, b) -> a + b * inputWeights[aInteger.getAndIncrement()][i]);
            hiddenOutputActivated[i] = sigmoidFunction(hiddenInputWeightSum[i]);
        });

        AtomicInteger integer = new AtomicInteger(0);
        outputSum = Arrays.stream(hiddenOutputActivated).reduce(0f, (a, b) -> a + b * outputWeights[integer.getAndIncrement()]);
        calculatedOutput = sigmoidFunction(outputSum);
        return calculatedOutput;
    }

    public void backward(float target, float[][] inputWeights, Float[] outputWeights, int neurons, Float[] inputs){
        float marginOfError = target - calculatedOutput;
        float deltaOutputSum = sigmoidFunctionDerivative(outputSum) * marginOfError;

        final AtomicInteger atomicInteger = new AtomicInteger(0);
        this.outputWeights = Arrays.stream(outputWeights).map(out -> {
            float deltaWeight = deltaOutputSum * hiddenOutputActivated[atomicInteger.getAndIncrement()];
            return out + deltaWeight;
        }).toArray(Float[]::new);

        float[] deltaHiddenSum = new float[neurons];
        float[][] deltaWeights = new float[inputs.length][neurons];
        IntStream.range(0, numOfNeurons).forEach(i -> {
            deltaHiddenSum[i] = deltaOutputSum * outputWeights[i] * (sigmoidFunctionDerivative(hiddenInputWeightSum[i]));
            IntStream.range(0, inputs.length).forEach(j -> {
                deltaWeights[j][i] = deltaHiddenSum[i] * inputs[j];
                this.inputWeights[j][i] = inputWeights[j][i] + deltaWeights[j][i];
            });
        });
    }

    private float sigmoidFunction(float value) {
        return (float)(1.0f/(1+Math.exp(-value)));
    }

    private float sigmoidFunctionDerivative(float value) {
        return sigmoidFunction(value) * (1-sigmoidFunction(value));
    }

    float[][] getInputWeights() {
        return inputWeights;
    }

    Float[] getOutputWeights() {
        return outputWeights;
    }
}
