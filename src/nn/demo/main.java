package nn.demo;
import nn.file.FileWriter;
import nn.testset.SetParser;

import java.util.ArrayList;
import java.util.Iterator;

import nn.testset.Vector;

import nn.NeuralNetwork;
import nn.file.FileParser;
import nn.utils.ActivationFunctions.*;
import nn.utils.Matrix.Matrix;
class Main {
public static void main(String[] args){
    long startTime = System.currentTimeMillis();

    double mean = 0;//72.9568306122449;
    double sd = 255;//89.96686299546113;
  
    NeuralNetwork network = new NeuralNetwork.Builder(4, 28*28)
        .addLayer(128, new ReLU())
        .addLayer(64, new ReLU())
        .addLayer(10, new SoftMax())
       .build();
    int batchSize = 32;
    double trainSetSize = 60000 * 0.8;
    double learningRate = 0.001;
    int epochs = 25;

    FileParser trainData = new FileParser("data/fashion_mnist_train_vectors.csv",sd, mean, 28*28);
    FileParser trainLabels = new FileParser("data/fashion_mnist_train_labels.csv",sd, mean,1);

    ArrayList<Vector> trainSet = new ArrayList<>();
    SetParser.parseTestSet(trainData, trainLabels, trainSet);
    ArrayList<Vector> trainSetEval = new ArrayList<>(trainSet);

    network.train(trainSet, trainSetSize, learningRate, epochs, batchSize);


    FileParser testData = new FileParser("data/fashion_mnist_test_vectors.csv",sd, mean, 28*28);
    
    FileWriter writer = new FileWriter("test_predictions.csv");
    FileParser testLabelsF = new FileParser("data/fashion_mnist_test_labels.csv",sd, mean, 1);
    int counter = 0;
    while(testData.hasNextVector()){
        double[] inputVector = testData.nextVector();
        double label = testLabelsF.nextDouble();
        network.setInput(inputVector);
        network.invoke();
        double[] outputVector = network.getOuput();
        int predictedLabel = Matrix.maxValueIndex(outputVector);
        writer.writeLabel(predictedLabel);
        if (predictedLabel == label){
            counter++;
        }
    }   

    writer.close();
    System.out.println("Test Accuracy: " + ((double)counter / 10000.0) * 100.0 + "%");

    writer = new FileWriter("train_predictions.csv");
    Iterator<Vector> trainEval = trainSetEval.iterator();
    counter = 0;
    while(trainEval.hasNext()){
        Vector vector = trainEval.next();
        network.setInput(vector.data);
        network.invoke();
        double[] outputVector = network.getOuput();
        int predictedLabel = Matrix.maxValueIndex(outputVector);
        writer.writeLabel(predictedLabel);
        if (predictedLabel == vector.label){
            counter++;
        }
    }
    writer.close();
    System.out.println("Train Accuracy: " + ((double)counter / 60000.00) * 100.0 + "%");

    long endTime = System.currentTimeMillis();
    System.out.println("Total time: " + (endTime - startTime)/1000.0 + " seconds");
}}

 