package nn.utils.Training;

public class AdamTrainingHelper extends TrainingHelper {

    double beta1;
    double beta2;
    double momentumMatrix[][][];
    double rmsMatrix[][][];
    public AdamTrainingHelper(double[][][] weights, double[][] outputs, nn.interfaces.ActivationFunction[] activationFunctions, double learningRate, int batchSize, double beta1, double beta2) {
        super(weights, outputs, activationFunctions, learningRate, batchSize);

        this.beta1 = beta1;
        this.beta2 = beta2;
        momentumMatrix = new double[weights.length][][];
        rmsMatrix = new double[weights.length][][];
        for(int layer =1; layer < weights.length; layer++){
            int numberOfNeurons = weights[layer].length;
            int numberOfWeights = weights[layer][0].length;
            momentumMatrix[layer] = new double[numberOfNeurons][numberOfWeights];
            rmsMatrix[layer] = new double[numberOfNeurons][numberOfWeights];
        }
        
    }
    

    @Override
    public void takeAStep(){
        double mHat;
        double rmsHat;
        for(int layer = 1; layer < weights.length; layer++){
            for(int neuron = 0; neuron < weights[layer].length; neuron++){
                for(int weight = 0; weight < weights[layer][neuron].length; weight++){
                    batchGradient[layer][neuron][weight] = batchGradient[layer][neuron][weight] / batchSize;

                    momentumMatrix[layer][neuron][weight] = beta1*momentumMatrix[layer][neuron][weight] + (1-beta1)*batchGradient[layer][neuron][weight];
                    mHat = momentumMatrix[layer][neuron][weight] /(1-beta1);

                    rmsMatrix[layer][neuron][weight] = beta2*rmsMatrix[layer][neuron][weight] + (1-beta2)*Math.pow(batchGradient[layer][neuron][weight],2);
                    rmsHat = rmsMatrix[layer][neuron][weight]/(1-beta2);
                    
                    weights[layer][neuron][weight] = weights[layer][neuron][weight] -(mHat/(Math.sqrt(rmsHat)+1e-8))*learningRate;
                    batchGradient[layer][neuron][weight] = 0;
                }
            }
        }
    }
    
}
