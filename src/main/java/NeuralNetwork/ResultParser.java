package NeuralNetwork;

public class ResultParser {

    private int nSuccess;
    private double threshold = 0.5;

    public void countSuccess(float calcForwardOutput, float target){
       if((calcForwardOutput < threshold && target == 0) || (calcForwardOutput >= (1 - threshold) && target == 1)){
           this.nSuccess += 1;
       }
    }

    public Integer parseResult(float result) {
        return (result < threshold) ? 0 : 1;
    }

    public int getNSuccess(){
        return this.nSuccess;
    }

    public void resetNSuccess() {
        this.nSuccess = 0;
    }
}
