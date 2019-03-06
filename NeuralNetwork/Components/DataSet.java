package NeuralNetwork.Components;

/**
 * DataSet -
 * The DataSet maintains one training
 * sample consisting of one input vector
 * and one output vector.
 *
 * @author Luca Jung
 * @version 3.4
 */
public class DataSet {

    private double[] inputVector;
    private double[] outputVector;

    public DataSet(){

    }

    public DataSet(double[] inputVector, double[] outputVector){
        this.inputVector = inputVector;
        this.outputVector = outputVector;
    }

    public void setInputVector(double[] inputVector) {
        this.inputVector = inputVector;
    }

    public void setOutputVector(double[] outputVector) {
        this.outputVector = outputVector;
    }

    public double[] getInputVector() {
        return inputVector;
    }

    public double[] getOutputVector() {
        return outputVector;
    }
}
