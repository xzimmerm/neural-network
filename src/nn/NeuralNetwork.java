package nn;
import nn.interfaces.Network; 
import nn.interfaces.ActivationFunction;
import nn.utils.Matrix.Matrix;

public class NeuralNetwork implements Network {
    
    private double[][][] weights;
    private double[][] potentials;
    private double[][] outputs;
    private ActivationFunction[] activationFunctions;

    public static class Builder implements Buildable<NeuralNetwork> {

        private double[][][] weights;
        private double[][] potentials;
        private double[][] outputs;
        private ActivationFunction[] activationFunctions;
        private int numberOfLayers;
        private int currLayer;

        public Builder(int numberOfLayers, int inputSize){
            currLayer = 0;
            this.numberOfLayers = numberOfLayers;

            weights = new double[numberOfLayers][][];
            potentials = new double[numberOfLayers][];
            outputs = new double[numberOfLayers][];
            activationFunctions = new ActivationFunction[numberOfLayers];

            outputs[currLayer] = new double[inputSize]; // allocate the input neurons
            currLayer++;

        }

        public Builder addLayer(int numberOfNeurons, ActivationFunction activationFunction){
            if (currLayer >= numberOfLayers){
                throw new IllegalStateException("Cannot add more layers than specified");
            }

            int prevLayerSize = outputs[currLayer - 1].length;

            weights[currLayer] = new double[numberOfNeurons][prevLayerSize + 1]; 
            potentials[currLayer] = new double[numberOfNeurons];
            outputs[currLayer] = new double[numberOfNeurons];
            activationFunctions[currLayer] = activationFunction;

            currLayer++;

            return this;
        }

        private void initializeWeights(){

            for(int layer = 1; layer < weights.length; layer++){
                for(int neuron = 0; neuron < weights[layer].length; neuron++){
                    for(int weight = 0; weight < weights[layer][neuron].length; weight++){
                        weights[layer][neuron][weight] = Math.random() * 2 - 1; // random value between -1 and 1
                    }
                }
            }
            
        }

        public NeuralNetwork build(){

            if (currLayer != numberOfLayers){
                throw new IllegalStateException("Not all layers have been added");
            }
            initializeWeights();
            NeuralNetwork network = new NeuralNetwork(weights, potentials, outputs, activationFunctions);

            return network;
        }
    }

    public NeuralNetwork(double[][][] weights, double[][] potentials, double[][] outputs, ActivationFunction[] activationFunctions){
        this.weights = weights;
        this.potentials = potentials; 
        this.outputs = outputs;
        this.activationFunctions = activationFunctions;
    }

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
        System.arraycopy(inputVector, 0, outputs[0], 0, inputVector.length);
    }

    private void resolveLayer(int layerNumber){
        
        double[][] layerWeightsMatrix = weights[layerNumber]; 
        double[] prevLayerOutputs = outputs[layerNumber - 1];
        double[] layerPotentials = potentials[layerNumber];
        double[] layerOutputs = outputs[layerNumber];
        ActivationFunction aFunction = activationFunctions[layerNumber];

        for(int neuron = 0; neuron < layerWeightsMatrix.length; neuron++){

            layerPotentials[neuron] = Matrix.weightProductAndSum(prevLayerOutputs, layerWeightsMatrix[neuron]);
            layerOutputs[neuron] = aFunction.activation(layerPotentials[neuron]);

        }
    }

}
