package nn.utils.ActivationFunctions;

import nn.interfaces.ActivationFunction;

public class SoftMax implements ActivationFunction{
    
    private double[] potentialsVector;
    private double sumExp; 


    public void activate(double[] potentialsVector){
        sumExp = 0;
        this.potentialsVector = potentialsVector;

        for(int neuron = 0; neuron < potentialsVector.length; neuron++){
            sumExp += Math.exp(potentialsVector[neuron]);
        }
    }

    public double activation(double input){

        return Math.exp(input)/sumExp;
    }

    public double derivation(double input){
        throw new IllegalStateException("Cannot get derivation of SoftMax without position");
    }

    

}
