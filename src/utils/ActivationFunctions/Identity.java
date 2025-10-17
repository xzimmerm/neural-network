package utils.ActivationFunctions;
import interfaces.ActivationFunction;

public class Identity implements ActivationFunction{

    public double activation(double input){
        return input;
    }

    public double derivation(double input){
        return 1;
    }
}
