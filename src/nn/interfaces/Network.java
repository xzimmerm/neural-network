package nn.interfaces;

import nn.file.FileParser;

public interface Network {

    public void train(FileParser dataFile, FileParser labelFile, int batchSize);

    public void invoke();

    public void setInput(double[] inputVector);

    public double[] getOuput();
}