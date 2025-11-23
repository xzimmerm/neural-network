package nn.utils.ActivationFunctions;

import nn.interfaces.ActivationFunction;

public class Dropout implements ActivationFunction{

    ActivationFunction originalAF; 
    double dropProbability;
    boolean trainingMode;

    public Dropout(ActivationFunction originalAF, double dropProbability, boolean trainingMode){
        this.originalAF = originalAF;
        this.dropProbability = dropProbability;
        this.trainingMode = trainingMode;
    }

    public double activation(double input){
        if(trainingMode){
            if(Math.random() < dropProbability){
            return 0;
        } else {
            return originalAF.activation(input);
        }
        } else{
            return originalAF.activation(input) * (1 - dropProbability);
        }
    }

    public double derivation(double input){
        return originalAF.derivation(input);
    }
    
    public void setTrainingMode(boolean trainingMode){
        this.trainingMode = trainingMode;
    }
}
