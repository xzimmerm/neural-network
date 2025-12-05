package nn.utils.ActivationFunctions;

import nn.interfaces.ActivationFunction;

/**
 * 
 */
public class SoftMax implements ActivationFunction{
    
    private double sumExp; 
    private double max;

    /**
     * Used to compute the sum needed for SoftMax activation
     * @param potentialsVector vector used to compute the sum
     */
    public void activate(double[] potentialsVector){
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
        return (Math.exp(input - max))/sumExp;
    }

    public double derivation(double input){
        
        return input * (1 - input);
    }

    

}
