package nn.threads;

public class InferenceTimeThread extends Thread {
    private int neuron; 
    private double[] weights;
    private double[] inputs;
    private double[] outputs;
    private double[] potentials;
    private nn.interfaces.ActivationFunction af;
    private double dropoutRate;
    

    public InferenceTimeThread(int neuron, double[] weights, double[] inputs, double[] outputs, double[] potentials, nn.interfaces.ActivationFunction af, double dropoutRate){
        this.neuron = neuron;
        this.weights = weights;
        this.inputs = inputs;
        this.outputs = outputs;
        this.potentials = potentials;
        this.af = af;
        this.dropoutRate = dropoutRate;
    }

    @Override
    public void run(){
        double potential = nn.utils.Matrix.Matrix.weightProductAndSum(inputs, weights);
        potentials[neuron] = potential;
        double activatedOutput = af.activation(potential);
        
        if (dropoutRate > 0) {
            activatedOutput *= (1.0 - dropoutRate);
        }
        
        outputs[neuron] = activatedOutput;
    }


    
}
