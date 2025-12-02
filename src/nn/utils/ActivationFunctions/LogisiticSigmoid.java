package nn.utils.ActivationFunctions;

public class LogisiticSigmoid implements nn.interfaces.ActivationFunction {

    public double activation(double input){
        return (1 / (1 + (Math.exp(-input))));
    }

    public double derivation(double input){
        double output = activation(input);
        return output * (1 - output);
    }
    
}
