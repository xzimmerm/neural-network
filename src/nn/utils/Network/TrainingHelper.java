package nn.utils.Network;

import nn.interfaces.ActivationFunction;

public class TrainingHelper {
    
    private double[][][] weights;
    private double[][] potentials;
    private double[][] outputs;
    private ActivationFunction[] activationFunctions;


    public TrainingHelper(double[][][] weights, double[][] potentials, double[][] outputs, ActivationFunction[] activationFunctions){
        this.weights = weights;
        this.potentials = potentials;
        this.outputs = outputs;
        this.activationFunctions = activationFunctions;
    }

    
}
