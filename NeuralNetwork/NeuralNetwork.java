package NeuralNetwork;

import NeuralNetwork.Components.DataSet;
import NeuralNetwork.Components.DataSets;
import NeuralNetwork.Components.MathTools;
import NeuralNetwork.Exceptions.NotReadyException;

/**
 * NeuralNetwork -
 * The neural network is capable
 * of learning through backpropagation.
 * It also supports batch learning and
 * different activation functions are given.
 *
 * @author Luca Jung
 * @version 2.2
 */
public class NeuralNetwork {

    private double[][]          networkNeurons;
    private double[][]          networkBias;
    private double[][][]        networkWeights;
    private double[][]          networkError;
    private boolean             isNetworkReady;
    private int                 activationFunction;

    /**
     * The constructor tries to set up
     * the neural network correctly.
     */
    public NeuralNetwork(int[] networkSize){
        if(isNetworkSizeCorrect(networkSize)) {
            //initialize network
            networkNeurons      = new double[networkSize.length][];
            networkBias         = new double[networkSize.length][];
            networkWeights      = new double[networkSize.length][][];
            networkError        = new double[networkSize.length][];

            //setting default values for each layer
            for (int i = 0; i < networkSize.length; i++) {
                networkNeurons[i]       = new double[networkSize[i]];
                networkError[i]         = new double[networkSize[i]];

                //setting random bias weights
                networkBias[i]          = new double[networkSize[i]];
                for (int j = 0; j < networkBias[i].length; j++) {
                    networkBias[i][j] = MathTools.getRandomDouble(-0.7, 0.7);
                }

                //setting random neuron weights
                if (i > 0) {
                    networkWeights[i] = new double[networkSize[i]][];
                    for (int j = 0; j < networkSize[i]; j++) {
                        networkWeights[i][j] = new double[networkSize[i - 1]];
                        for (int k = 0; k < networkWeights[i][j].length; k++) {
                            networkWeights[i][j][k] = MathTools.getRandomDouble(-0.7, 0.7);
                        }
                    }
                }
            }

            //using Sigmoid as default activation function
            activationFunction = MathTools.SIGMOID_FUNCTION;

            //network is now ready to use
            isNetworkReady = true;
        }
        else {
            isNetworkReady = false;
            throw new IllegalArgumentException("Can't create network from given parameters!");
        }

    }

    //###### Start - prediction Methods ######

    /**
     * The prediction process is made up of two parts:
     * 1. setting the input vector
     * 2. forwardpass
     */
    public double[] predict(double[] inputVector){
        if(isNetworkReady()) {
            if(inputVector.length == inputSize()) {
                //setting input vector
                setInputVector(inputVector);

                //compute result
                forwardPass();

                //return output layer
                return getOutputLayer();
            }
            else {
                throw new IllegalArgumentException("The given input vector doesn't match the corresponding network input size");
            }
        }
        else {
            throw new NotReadyException("The network wasn't set up correct and is not ready to work!");
        }
    }

    /**
     * The forwardpass calculates the output
     * based on the current input vector.
     */
    private void forwardPass(){
        for (int layer = 1; layer < networkSize(); layer++) {
            for (int neuron = 0; neuron < size(layer); neuron++) {
                double sum = networkBias[layer][neuron];
                for (int previousLayerNeuron = 0; previousLayerNeuron < size(layer - 1); previousLayerNeuron++) {
                    sum += networkNeurons[layer - 1][previousLayerNeuron] * networkWeights[layer][neuron][previousLayerNeuron];
                }
                networkNeurons[layer][neuron] = MathTools.useActivationFunction(activationFunction, sum, false);
            }
        }
    }


    /**
     * Setting the input vector as output
     * of the first layer.
     */
    private void setInputVector(double[] inputVector){
        for (int i = 0; i < inputSize(); i++) {
            networkNeurons[0][i] = inputVector[i];
        }
    }

    //###### End - prediction Methods ######


    //###### Start - training Methods ######

    /**
     * Basic training methods:
     */
    public void startTraining(DataSets dataSets, int iterations, double learningRate, int batchSize){
        if(isNetworkReady()) {
            for (int i = 0; i < iterations; i++) {
                DataSets workingDataSets = dataSets.getRandomSubSet(batchSize);
                for (int j = 0; j < workingDataSets.size(); j++) {
                    DataSet currentDataSet = workingDataSets.getDataSet(j);
                    startTraining(currentDataSet.getInputVector(), currentDataSet.getOutputVector(), learningRate);
                }
            }
        }
        else {
            throw new NotReadyException("The network wasn't set up correct and is not ready to work!");
        }
    }

    public void startTraining(DataSet dataSet, double learningRate){
        if(isNetworkReady()) {
            startTraining(dataSet.getInputVector(), dataSet.getOutputVector(), learningRate);
        }
        else {
            throw new NotReadyException("The network wasn't set up correct and is not ready to work!");
        }
    }

    private void startTraining(double[] inputVector, double[] targetVector, double learningRate){
        predict(inputVector);
        calculateError(targetVector);
        applyWeightChanges(learningRate);
    }

    /**
     * Calculating the error.
     */
    public void calculateError(double[] targetVector){
        for (int i = 0; i < size(lastLayerIndex()); i++) {
            networkError[lastLayerIndex()][i] = MathTools.useActivationFunction(activationFunction ,networkNeurons[lastLayerIndex()][i], true) * (networkNeurons[lastLayerIndex()][i] - targetVector[i]);
        }

        for (int layer = lastLayerIndex() - 1; layer > 0; layer--) {
            for (int neuron = 0; neuron < size(layer); neuron++) {
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < size(layer + 1); nextNeuron++) {
                    sum += networkWeights[layer + 1][nextNeuron][neuron] * networkError[layer + 1][nextNeuron];
                }
                networkError[layer][neuron] = sum * MathTools.useActivationFunction(activationFunction , networkNeurons[layer][neuron], true);
            }
        }
    }

    /**
     * Add the weight change to the weights.
     */
    public void applyWeightChanges(double learningRate){
        for (int layer = 1; layer < networkSize(); layer++) {
            for (int neuron = 0; neuron < size(layer); neuron++) {
                for (int previousLayerNeuron = 0; previousLayerNeuron < size(layer - 1); previousLayerNeuron++) {
                    double delta = -1 * learningRate * networkNeurons[layer - 1][previousLayerNeuron] * networkError[layer][neuron];
                    networkWeights[layer][neuron][previousLayerNeuron] += delta;
                }
                double delta = -1 * learningRate * networkError[layer][neuron];
                networkBias[layer][neuron] += delta;
            }
        }
    }

    //###### End - training Methods ######

    //###### Start - helper Methods ######

    private boolean isNetworkSizeCorrect(int[] networkSize){
        if(networkSize.length > 1){
            for (int i = 0; i < networkSize.length; i++) {
                if(networkSize[i] < 1){
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    public void setActivationFunction(int activationFunction){
        this.activationFunction = activationFunction;
    }

    public boolean isNetworkReady(){
        return isNetworkReady;
    }

    public int inputSize(){
        return networkNeurons[0].length;
    }

    public int networkSize(){
        return networkNeurons.length;
    }

    public int lastLayerIndex(){
        return networkSize() - 1;
    }

    public int size(int layer){
        return networkNeurons[layer].length;
    }

    public int outputSize(){
        return networkNeurons[networkSize() - 1].length;
    }

    private double[] getOutputLayer(){
        return networkNeurons[lastLayerIndex()];
    }

    //###### End - helper Methods ######

    public String toString(){
        String string = "A neural network with " + networkSize() + " layer(s) including a bias.\nUsing activation function: " + MathTools.getActivationFunctionName(activationFunction) + ".\nNetwork state: ";
        if(isNetworkReady()){
            string += "ready.";
        }
        else {
            string += "not ready.";
        }
        return string;
    }

}
