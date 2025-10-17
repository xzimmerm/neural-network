package nn.utils.ActivationFunctions ; 
import nn.interfaces.ActivationFunction;


public final class ReLU implements ActivationFunction {

    public double activation(double input){

        return Math.max(input, 0);
    }

    public double derivation(double input){

        if (input >= 0){
            return 1;
        } else {
            return 0;
        }
    }
}