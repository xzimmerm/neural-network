package nn;
import nn.interfaces.Network;
import nn.testset.Vector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;

import nn.interfaces.ActivationFunction;
import nn.utils.Matrix.Matrix;
import nn.utils.Training.AdamTrainingHelper;
import nn.utils.Training.TrainingHelper;
import nn.utils.ActivationFunctions.Identity;
import nn.utils.ActivationFunctions.SoftMax;
public class NeuralNetwork implements Network {
    
    private double[][][] weights;
    private double[][] potentials;
    private double[][] outputs;
    private ActivationFunction[] activationFunctions;
    private TrainingHelper trainingHelper = null;
    private String dropoutMode = null;
    private double[] dropoutRates;
    private boolean[][] dropoutMasks;


    public static class Builder implements Buildable<NeuralNetwork> {

        private double[][][] weights;
        private double[][] potentials;
        private double[][] outputs;
        private ActivationFunction[] activationFunctions;
        private int numberOfLayers;
        private int currLayer;
        private double[] dropoutRates;
        private boolean[][] dropoutMasks;
        private String dropoutMode = null; 

        public Builder(int numberOfLayers, int inputSize){
            currLayer = 0;
            this.numberOfLayers = numberOfLayers;

            weights = new double[numberOfLayers][][];
            potentials = new double[numberOfLayers][];
            outputs = new double[numberOfLayers][];
            activationFunctions = new ActivationFunction[numberOfLayers];
            dropoutMasks = new boolean[numberOfLayers][];
            dropoutRates = new double[numberOfLayers];

            outputs[currLayer] = new double[inputSize]; // allocate the input neurons
            potentials[currLayer] = outputs[currLayer]; // just to have something there and be consistent
            activationFunctions[currLayer] = new Identity(); // same as above
            weights[0] =  null; // no weights for input layer 
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

        public Builder addDropout(double dropoutRate){

            dropoutMode = "train";
            dropoutRates[currLayer - 1] = dropoutRate;
            dropoutMasks[currLayer - 1] = new boolean[outputs[currLayer - 1].length];
            return this;
        }

        private void initializeWeights(){

            for(int layer = 1; layer < weights.length; layer++){
                Random r = new Random();
                int prevLayerSize = outputs[layer - 1].length;
                for(int neuron = 0; neuron < weights[layer].length; neuron++){
                    for(int weight = 0; weight < weights[layer][neuron].length; weight++){
                        weights[layer][neuron][weight] = r.nextGaussian()*(2)/(prevLayerSize); // random value between -1 and 1
                    }
                }
            }
        }

        public NeuralNetwork build(){

            if (currLayer != numberOfLayers){
                throw new IllegalStateException("Not all layers have been added");
            }
            initializeWeights();
            NeuralNetwork network = new NeuralNetwork(weights, potentials, outputs, activationFunctions, dropoutMode, dropoutRates, dropoutMasks);

            
            return network;
        }
    }

    public NeuralNetwork(double[][][] weights, double[][] potentials, double[][] outputs, ActivationFunction[] activationFunctions, String dropoutMode, double[] dropoutRates, boolean[][] dropoutMasks){ 
        this.weights = weights;
        this.potentials = potentials; 
        this.outputs = outputs;
        this.activationFunctions = activationFunctions;
        this.dropoutMode = dropoutMode;
        this.dropoutRates = dropoutRates;
        this.dropoutMasks = dropoutMasks;
    }

    public double[] getOuput(){
        return outputs[outputs.length - 1];
    }

    public void invoke(){
        for(int layer = 1; layer < outputs.length; layer++){
            if(dropoutMode != null && dropoutRates[layer] > 0){
                resolveDropoutLayer(layer);
            }
            else{
                resolveLayer(layer);
            }
        }
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

    
        if (aFunction instanceof SoftMax){
            for (int neuron = 0; neuron < layerWeightsMatrix.length; neuron++){
                layerPotentials[neuron] = Matrix.weightProductAndSum(prevLayerOutputs, layerWeightsMatrix[neuron]);
            }

            ((SoftMax)aFunction).activate(layerPotentials);
            
            for (int neuron = 0; neuron < layerWeightsMatrix.length; neuron++){
                layerOutputs[neuron] = aFunction.activation(layerPotentials[neuron]);
            }
        }
        else{
            for(int neuron = 0; neuron < layerWeightsMatrix.length; neuron++){
                layerPotentials[neuron] = Matrix.weightProductAndSum(prevLayerOutputs, layerWeightsMatrix[neuron]);
                layerOutputs[neuron] = aFunction.activation(layerPotentials[neuron]);
            }
        }
    }

    private void resolveDropoutLayer(int layerNumber){

            double layerPotentials[] = potentials[layerNumber];
            double layerOutputs[] = outputs[layerNumber];
            ActivationFunction aFunction = activationFunctions[layerNumber];
            double[][] layerWeightsMatrix = weights[layerNumber];
            double[] prevLayerOutputs = outputs[layerNumber - 1];

        if (dropoutMode.equals("inference")){
            for(int neuron = 0; neuron < outputs[layerNumber].length; neuron++){
                layerPotentials[neuron] = Matrix.weightProductAndSum(prevLayerOutputs, layerWeightsMatrix[neuron]) *(double) (1.0 - dropoutRates[layerNumber]);
                layerOutputs[neuron] = aFunction.activation(layerPotentials[neuron]);
            
            }
        } else{
            for(int neuron = 0; neuron < layerWeightsMatrix.length; neuron++){
                if(dropoutMasks[layerNumber][neuron]){
                    layerPotentials[neuron] = 0;
                    layerOutputs[neuron] = 0;
                }
                else{
                layerPotentials[neuron] = Matrix.weightProductAndSum(prevLayerOutputs, layerWeightsMatrix[neuron]);
                layerOutputs[neuron] = aFunction.activation(layerPotentials[neuron]);
                }
            }
        }
    }

    private void changeDropoutMode(boolean trainingMode){
        if(trainingMode){
            dropoutMode = "train";
           // setDropoutMode();
        } else {
            dropoutMode = "inference";
        }
    }

    private void setDropoutMasks(){
        Random r = new Random();
        for(int layer = 1; layer < outputs.length; layer++){
            if(dropoutRates[layer] > 0){
                for(int neuron = 0; neuron < outputs[layer].length; neuron++){
                    double prob = r.nextDouble();
                    if (prob < dropoutRates[layer]){
                        dropoutMasks[layer][neuron] = true; // drop this neuron
                    } else {
                        dropoutMasks[layer][neuron] = false; // keep this neuron
                    }
                }
        }
        }
    }

    private void trainingEpoch(ArrayList<Vector> trainSet, double learningRate, int epoch, int batchSize, double trainSetSize){
        
        double numberOfBatches = trainSetSize / batchSize;
        Collections.shuffle(trainSet.subList(0,(int)trainSetSize));
        double crossEntropyEpoch = 0;
        double batch = 0;
        Iterator<Vector> trainSetIterator = trainSet.iterator();
        while(batch < numberOfBatches){
            if (dropoutMode != null){
                setDropoutMasks();
            }
            double crossEntropy = 0;
            int batchVector = 0;
            while(batchVector < batchSize){
                Vector vector = trainSetIterator.next();
                setInput(vector.data);
                invoke();
                trainingHelper.backpropagate(vector.label, learningRate);
                crossEntropy += -Math.log(outputs[outputs.length - 1][(int)vector.label] + 1e-15);
                batchVector++;
            }
            System.out.println("Average cross entropy for batch " + (batch + 1) + ": " + (crossEntropy / batchSize));
            crossEntropyEpoch += crossEntropy / batchSize;
            trainingHelper.takeAStep();
            batch++;
            
        }
        System.out.println("Average cross entropy for epoch " + (epoch + 1) + ": " + crossEntropyEpoch/numberOfBatches);
         
        validationEpoch(trainSetIterator);
    }

    private void validationEpoch(Iterator<Vector> valSetIterator){
        int correctPredictions = 0;
        
        if (dropoutMode != null){
            changeDropoutMode(false); // disable dropout for validation
        }
        int vectorCount = 0;
        Vector valVector;
        while(valSetIterator.hasNext()){
            valVector = valSetIterator.next();
            vectorCount++;
            setInput(valVector.data);
            invoke();
            int predictedLabel = Matrix.maxValueIndex(outputs[outputs.length - 1]);
            if (predictedLabel == valVector.label){
                correctPredictions++;
            }
            }
        
        System.out.println("Validation accuracy: " + ((double)correctPredictions / vectorCount) * 100.0 + "%");

        if(dropoutMode != null){
            changeDropoutMode(true); // enable dropout back for training
        }
    } 
    public void train(ArrayList<Vector> trainSet, double trainSetSize, double learningRate, int epochs, int batchSize){
        
        if (trainingHelper == null){
            trainingHelper = new AdamTrainingHelper(weights, outputs, potentials, activationFunctions, learningRate, batchSize, 0.9f, 0.999f);
        }
        if (dropoutMode != null){
            changeDropoutMode(true);
        }

        for(int epoch = 0;epoch < epochs; epoch++){
            System.out.println("Epoch number: " + (epoch + 1));
            trainingEpoch(trainSet, learningRate, epoch, batchSize, trainSetSize);
        }

        if(dropoutMode != null){
            changeDropoutMode(false); // disable dropout after training
        }
    }
}
