package utils.ActivationFunctions ; 
import interfaces.ActivationFunction;


public final class ReLU implements ActivationFunction {

    public float activation(float input){

        return Math.max(input, 0);
    }

    public float derivation(float input){

        if (input >= 0){
            return 1;
        } else {
            return 0;
        }
    }
}