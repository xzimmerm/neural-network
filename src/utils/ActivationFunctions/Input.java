package utils.ActivationFunctions;
import interfaces.ActivationFunction;

public final class Input implements ActivationFunction{

    public float activation(float input){
        return input;
    }

    public float derivation(float input){
        return 1;
    }
}
