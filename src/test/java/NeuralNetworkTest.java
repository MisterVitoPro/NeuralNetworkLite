import neuralNetwork.NeuralNetwork;
import neuralNetwork.Propagator;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NeuralNetworkTest {

    @Test
    public void ForwardPropagateWithSigmoidTest() {
        final int hiddenNeurons = 3;
        final Float[][] inputs = new Float[][]{
                {1f,1f}
        };
        final Integer[] outputs = new Integer[]{0};

        final float[][] inputWeights = new float[][]{
                {0.8f, 0.4f, 0.5f},
                {0.4f, 0.9f, 0.3f},
        };
        final Float[] outputWeights = new Float[]{0.3f, 0.5f, 0.9f};

        final NeuralNetwork neuralNetwork = new NeuralNetwork(inputs, outputs);
        neuralNetwork.addHiddenLayer(hiddenNeurons);
        System.out.println("Predicted result for [1,1]: " + neuralNetwork.predict(new Float[] {1f, 1f}));

        Propagator propagator = new Propagator(inputWeights, outputWeights, hiddenNeurons);
        final float calcForwardOutput = propagator.forward(inputs[0].clone());
        Assert.assertEquals(calcForwardOutput, 0.776338f);
    }

}
