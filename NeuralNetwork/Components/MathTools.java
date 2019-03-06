package NeuralNetwork.Components;

/**
 * MathTools -
 * MathTools is a collection of all activation
 * functions and their derivatives including
 * some basic operations to get random values.
 *
 * @author Luca Jung
 * @version 2.0
 */
public class MathTools {

    public static final int IDENTITY_FUNCTION = 0;
    public static final int SIGMOID_FUNCTION = 1;
    public static final int RELU_FUNCTION = 2;
    public static final int LEAKYRELU_FUNCTION = 3;
    public static final int TANGENS_HYPERBOLICUS_FUNCTION  = 4;

    public static double useActivationFunction(int activationFunction, double x, boolean derivative){
        switch (activationFunction){
            case IDENTITY_FUNCTION:
                return identityFunction(x, derivative);

            case SIGMOID_FUNCTION:
                return sigmoidFunction(x, derivative);

            case RELU_FUNCTION:
                return reLUFunction(x, derivative);

            case LEAKYRELU_FUNCTION:
                return leakyReLUFunction(x, derivative);

            case TANGENS_HYPERBOLICUS_FUNCTION:
                return tangensHyperbolicusFunction(x, derivative);

            default:
                return x;
        }
    }

    public static String getActivationFunctionName(int activationFunction){
        switch (activationFunction){
            case IDENTITY_FUNCTION:
                return "Identity function";

            case SIGMOID_FUNCTION:
                return "Sigmoid function";

            case RELU_FUNCTION:
                return "ReLU function";

            case LEAKYRELU_FUNCTION:
                return "Leaky ReLU function";

            case TANGENS_HYPERBOLICUS_FUNCTION:
                return "Tangens hyperbolicus function (tanh)";

            default:
                return "function invalid";
        }
    }

    private static double sigmoidFunction(double x, boolean derivative){
        if(!derivative) {
            return 1 / (1 + Math.exp(-x));
        }
        else {
            return sigmoidFunction(x, false) * (1 - sigmoidFunction(x, false));
        }
    }

    private static double reLUFunction(double x, boolean derivative){
        if(!derivative) {
            if (x > 0) {
                return x;
            } else {
                return 0.0;
            }
        }
        else {
            if(x > 0){
                return 1.0;
            }
            else {
                return 0.0;
            }
        }
    }

    private static double leakyReLUFunction(double x, boolean derivative){
        if(!derivative) {
            if (x > 0) {
                return x;
            } else {
                return 0.01 * x;
            }
        }
        else {
            if(x > 0){
                return 1.0;
            }
            else {
                return 0.01;
            }
        }
    }

    private static double identityFunction(double x, boolean derivative){
        if(!derivative) {
            return x;
        }
        else {
            return 1.0;
        }
    }

    private static double tangensHyperbolicusFunction(double x, boolean derivative){
        if(!derivative) {
            return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
        }
        else {
            return 1 - (tangensHyperbolicusFunction(x, false) * tangensHyperbolicusFunction(x, false));
        }
    }

    public static double getRandomDouble(double min, double max){
        return min + Math.random() * (max - min);
    }

    public static int getRandomInt(int min, int max){
        return min + (int)(Math.random() * ((max - min) + 1));
    }

}
