package nn.utils.ActivationFunctions;

import nn.interfaces.ActivationFunction;

public class SoftMax implements ActivationFunction{
    
    private double[] potentialsVector;
    private double sumExp; 
    private double max;


    public void activate(double[] potentialsVector){
        this.potentialsVector = potentialsVector;
        max = potentialsVector[0];
        for(int neuron = 1; neuron < potentialsVector.length; neuron++){
            if(potentialsVector[neuron] > max){
                max = potentialsVector[neuron];
            }
        }
        sumExp = 0;
        for(int neuron = 0; neuron < potentialsVector.length; neuron++){
            sumExp += Math.exp(potentialsVector[neuron] - max);
        }
    }

    public double activation(double input){
        return Math.exp(input - max)/sumExp;
    }

    public double derivation(double input){
        
        return input * (1 - input);
    }

    

}
