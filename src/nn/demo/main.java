package nn.demo;
import nn.file.FileWriter;

import javax.management.relation.RelationException;

import nn.NeuralNetwork;
import nn.file.FileParser;
import nn.utils.ActivationFunctions.*;
import nn.utils.Matrix.Matrix;
class Main {
public static void main(String[] args){
    
    FileParser trainData = new FileParser("data/fashion_mnist_train_vectors.csv", 28*28);
    FileParser trainLabels = new FileParser("data/fashion_mnist_train_labels.csv", 1);
    //int i = 0 ;
    //while(parser.hasNextVector()){
        //parser.nextVector();
        //System.out.println(i);
        //i++;
    //}
    double sum = 0;
    double mean = 72.9568306122449;
    double sd = 89.96686299546113;
    // while(trainData.hasNextVector()){
    //     double[] inputVector = trainData.nextVector();
    //     for(int i = 0; i < inputVector.length; i++){
    //         sum += Math.pow(inputVector[i] - mean,2); // normalize input
    //     }
    // }

    System.out.println("sd: " + Math.sqrt(sum/(60000*28*28)));   
    NeuralNetwork network = new NeuralNetwork.Builder(9, 28*28)
    .addLayer(392, new Identity())
    .addLayer(392, new ReLU())
    .addDropout(0.25)
    .addLayer(196, new Identity())
       .addLayer(196, new ReLU())
       .addDropout(0.25)
       .addLayer(98, new Identity())
       .addLayer(98, new ReLU())
       .addDropout(0.25)
       .addLayer(10, new Identity())
       .addLayer(10, new SoftMax())
       .build();
    
    network.train(trainData, trainLabels, 100, 0, 1);
    //for(int i = 0; i < newVector.length; i++){
     // ##  System.out.println(newVector[i]);
   // }

    FileParser testData = new FileParser("data/fashion_mnist_test_vectors.csv", 28*28);
    
    FileWriter writer = new FileWriter("data/fashion_mnist_test_predictions.csv");
    FileParser testLabels = new FileParser("data/fashion_mnist_test_labels.csv", 1);
    int counter = 0;
    while(testData.hasNextVector()){
        double[] inputVector = testData.nextVector();
        for(int i = 0; i < inputVector.length; i++){
                   inputVector[i] = (inputVector[i]-mean) / sd; // normalize input
        }
        double label = testLabels.nextDouble();
        network.setInput(inputVector);
        network.invoke();
        double[] outputVector = network.getOuput();
        int predictedLabel = Matrix.maxValueIndex(outputVector);
        double val = outputVector[predictedLabel];
        System.out.println("Vector no. " + counter +"predicted label: " + predictedLabel + "value:" + val);
        writer.writeLabel(predictedLabel);
        if (predictedLabel == (int)label){
            counter++;
        }
}

writer.close();
System.out.println("Accuracy: " + ((double)counter / 10000.0) * 100.0 + "%");
}}