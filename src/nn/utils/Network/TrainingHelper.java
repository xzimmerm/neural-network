package nn.utils.Network;

import nn.interfaces.ActivationFunction;
import nn.utils.Matrix.Matrix;

public class TrainingHelper {
    
    private double[][][] weights;
    private double[][] potentials;
    private double[][] outputs;
    private ActivationFunction[] activationFunctions;
    private double[][][] batchGradient; 
    private double[][][] vectorGradient;

    public TrainingHelper(double[][][] weights, double[][] potentials, double[][] outputs, ActivationFunction[] activationFunctions){
        this.weights = weights;
        this.potentials = potentials;
        this.outputs = outputs;
        this.activationFunctions = activationFunctions;
    }

    public void backpropagate(double correctLabel){



        Matrix.addAndZeroOutMatrix(batchGradient, vectorGradient);
    }

    public void takeAStep(){

        Matrix.zeroOutMatrix(batchGradient);
    }

    
}
