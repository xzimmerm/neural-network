package nn.interfaces;

import java.util.ArrayList;

import nn.testset.Vector;

public interface Network {

    public void train(ArrayList<Vector> trainSet, double trainSetSize, double learningRate, int epochs, int batchSize);

    public void invoke();

    public void setInput(double[] inputVector);

    public double[] getOuput();
}