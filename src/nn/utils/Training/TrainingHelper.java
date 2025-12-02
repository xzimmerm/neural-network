package nn.utils.Training;

import nn.interfaces.ActivationFunction;
import nn.utils.Matrix.Matrix;

public class TrainingHelper {
    
    protected double[][][] weights;
    private double[][] outputs;
    protected double[][] potentials;
    private ActivationFunction[] activationFunctions;
    protected double[][][] batchGradient; 
    private double[][] deltas;
    protected double learningRate;
    protected int batchSize;

    public TrainingHelper(double[][][] weights, double[][] outputs, double[][] potentials, ActivationFunction[] activationFunctions, double learningRate, int batchSize){
        this.weights = weights;
        this.outputs = outputs;
        this.potentials = potentials;
        this.activationFunctions = activationFunctions;
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        
        initArrays();
    }

    private void initArrays(){
        batchGradient = new double[weights.length][][];
        deltas = new double[outputs.length][];

        for(int layer = 1; layer < batchGradient.length; layer++){
            int numberOfNeurons = weights[layer].length;
            int numberOfWeights = weights[layer][0].length;

            batchGradient[layer] = new double[numberOfNeurons][numberOfWeights];
            deltas[layer] = new double[numberOfNeurons];
        }
    }


    public void backpropagate(double correctLabel, double learningRate){
    this.learningRate = learningRate;

    computeDeltas(correctLabel);
    computeBatchGradient();

       
    }

    private void computeDeltas(double correctLabel){
          outputLayerDelta(correctLabel);
        for(int layer = deltas.length - 2; layer > 0; layer--){
            hiddenLayerDelta(layer);
        }
    }

    private void outputLayerDelta(double correctLabel){
        ActivationFunction af = activationFunctions[activationFunctions.length - 1];
        double[] deltasLayer = deltas[deltas.length - 1];
        double output;
        double error;
        double classification;
        for(int neuron = 0; neuron < deltas[deltas.length - 1].length; neuron++){
            output = outputs[outputs.length - 1][neuron];
            if (correctLabel == neuron){
                deltasLayer[neuron] = output - 1;//*(af.derivation(output));
            } else {
                deltasLayer[neuron] = output;//*(af.derivation(output));
            }
        }
    }

    private void hiddenLayerDelta(int layer){
        ActivationFunction af = activationFunctions[layer];
        double[] deltasLayer = deltas[layer];
        double[] outputsLayer = outputs[layer];
        double[] nextLayerDeltas = deltas[layer + 1];
        double[][] nextLayerWeights = weights[layer + 1];
        double[] potentialsLayer = potentials[layer];
        double sum;

        for(int neuron = 0; neuron < deltasLayer.length; neuron++){
            sum = 0;
            if(af.derivation(potentialsLayer[neuron]) != 0){
                for(int nextNeuron = 0; nextNeuron < nextLayerDeltas.length; nextNeuron++){
                    sum += nextLayerWeights[nextNeuron][neuron + 1] * nextLayerDeltas[nextNeuron];
                }
                deltasLayer[neuron] = sum * af.derivation(potentialsLayer[neuron]);
            } else {
                deltasLayer[neuron] = 0;
            }
        }
    }
    
    private void computeBatchGradient(){
        for(int layer = 1; layer < weights.length; layer++){
            double[] deltasLayer = deltas[layer];
            double[] outputsPrevLayer = outputs[layer - 1];
            double[][] batchGradientLayer = batchGradient[layer];

            for(int neuron = 0; neuron < batchGradientLayer.length; neuron++){
                batchGradientLayer[neuron][0] += deltasLayer[neuron]; // bias weight
                for(int weight = 1; weight < batchGradientLayer[neuron].length; weight++){
                    boolean check = Double.isNaN(batchGradientLayer[neuron][weight]);
                    batchGradientLayer[neuron][weight] += deltasLayer[neuron] * outputsPrevLayer[weight - 1];
                    
                }
            }
        }
    }

    public void takeAStep(){
        
        Matrix.addAndZeroOutMatrix(weights, batchGradient, batchSize, learningRate);
    }


    
}
