package nn;
import nn.interfaces.Network; 
import nn.interfaces.ActivationFunction;
import nn.utils.Matrix.Matrix;

public class NeuralNetwork implements Network {
    
    private double[][][] weights;
    private double[][] potentials;
    private double[][] outputs;
    private ActivationFunction[] activationFunctions;

    public double[] getOuput(){
        return outputs[outputs.length - 1];
    }

    public void invoke(){
        for(int layer = 1; layer < outputs.length; layer++){
            resolveLayer(layer);
        }
    }

    public void train(){

        return;
    }

    public void setInput(double[] inputVector){
        System.arraycopy(inputVector, 0, potentials[0], 0, inputVector.length);
    }

    private void resolveLayer(int layerNumber){
        
        double[][] layerWeightsMatrix = weights[layerNumber]; 
        double[] prevLayerOutputs = outputs[layerNumber - 1];
        double[] layerPotentials = potentials[layerNumber];
        double[] layerOutputs = outputs[layerNumber];
        ActivationFunction aFunction = activationFunctions[layerNumber];

        for(int neuron = 0; neuron < layerWeightsMatrix.length; neuron++){

            layerPotentials[neuron] = Matrix.arrayProductAndSum(prevLayerOutputs, layerWeightsMatrix[neuron]);
            layerOutputs[neuron] = aFunction.activation(layerPotentials[neuron]);

        }
    }

}
