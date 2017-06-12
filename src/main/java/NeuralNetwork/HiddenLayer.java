package NeuralNetwork;

import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

/**
 * Creates the weights from inputs-to-hidden and hidden-to-outputs
 */
class HiddenLayer {

    private Float[] outputWeights;
    private float[][] inputWeights;
    private int neurons;

    HiddenLayer(int neurons, int inputs) {
        this.neurons = neurons;
        this.outputWeights = new Float[neurons];
        this.inputWeights = new float[inputs][neurons];

        IntStream.range(0, neurons).forEach(i -> {
            this.outputWeights[i] = ThreadLocalRandom.current().nextFloat();
            IntStream.range(0, inputWeights.length).forEach(j -> this.inputWeights[j][i] = ThreadLocalRandom.current().nextFloat());
        });
    }

    Float[] getOutputWeights(){
        return this.outputWeights;
    }

    float[][] getInputWeights(){
        return this.inputWeights;
    }

    public int getNeurons() {
        return neurons;
    }

    public void setOutputWeights(Float[] outputWeights) {
        this.outputWeights = outputWeights;
    }

    public void setInputWeights(float[][] inputWeights) {
        this.inputWeights = inputWeights;
    }
}
