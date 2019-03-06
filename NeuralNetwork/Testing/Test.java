package NeuralNetwork.Testing;

import NeuralNetwork.Components.DataSet;
import NeuralNetwork.Components.DataSets;
import NeuralNetwork.NeuralNetwork;

import java.util.Arrays;

public class Test {

    public static void main(String[] args){
        NeuralNetwork neuralNetwork = new NeuralNetwork(new int[]{2,3,1});
        System.out.println(neuralNetwork.toString());
        System.out.println();

        DataSets dataSets = new DataSets();
        DataSet dataSet1 = new DataSet(new double[]{0.5,0.4}, new double[]{0.9});
        DataSet dataSet2 = new DataSet(new double[]{0.4,-0.4}, new double[]{0.0});
        dataSets.addDataSet(dataSet1);
        dataSets.addDataSet(dataSet2);

        System.out.println("Initial network: ");
        System.out.println("Prediction: " + Arrays.toString(neuralNetwork.predict(dataSet1.getInputVector())) + " <--> expected: " +  Arrays.toString(dataSet1.getOutputVector()));
        System.out.println("Prediction: " + Arrays.toString(neuralNetwork.predict(dataSet2.getInputVector())) + " <--> expected: " +  Arrays.toString(dataSet2.getOutputVector()));
        System.out.println();

        neuralNetwork.startTraining(dataSets, 10000, 0.2, 1);

        System.out.println("Trained network: ");
        System.out.println("Prediction: " + Arrays.toString(neuralNetwork.predict(dataSet1.getInputVector())) + " <--> expected: " +  Arrays.toString(dataSet1.getOutputVector()));
        System.out.println("Prediction: " + Arrays.toString(neuralNetwork.predict(dataSet2.getInputVector())) + " <--> expected: " +  Arrays.toString(dataSet2.getOutputVector()));
    }

}
