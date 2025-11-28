package nn.demo;
import nn.file.FileWriter;

import nn.NeuralNetwork;
import nn.file.FileParser;
import nn.utils.ActivationFunctions.*;
import nn.utils.Matrix.Matrix;
class Main {
public static void main(String[] args){
    
    //int i = 0 ;
    //while(parser.hasNextVector()){
        //parser.nextVector();
        //System.out.println(i);
        //i++;
    //}
    double sum = 0;
    double mean = 0.0;//72.9568306122449;
    double sd = 255.0;//89.96686299546113;
    // while(trainData.hasNextVector()){
    //     double[] inputVector = trainData.nextVector();
    //     for(int i = 0; i < inputVector.length; i++){
    //         sum += Math.pow(inputVector[i] - mean,2); // normalize input
    //     }
    // }

    System.out.println("sd: " + Math.sqrt(sum/(60000*28*28)));   
    NeuralNetwork network = new NeuralNetwork.Builder(4, 28*28)
        .addLayer(200, new ReLU())
        .addDropout(0.25)
        .addLayer(64, new ReLU())
        .addDropout(0.25)
        .addLayer(10, new SoftMax())
       .build();
    
    double learningRate = 0.001;
    for (int epoch = 0; epoch < 20; epoch++){
    // if (learningRate > 0.01 && epoch >=5 ){
    //     learningRate = learningRate - 0.01;
    // }
    System.out.println("Epoch number: " + (epoch+1));
    FileParser trainData = new FileParser("data/fashion_mnist_train_vectors.csv", 28*28);
    FileParser trainLabels = new FileParser("data/fashion_mnist_train_labels.csv", 1);
    network.train(trainData, trainLabels, 32, mean, sd, learningRate);
    } 
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

    // public static void main(String[] args) {
        
    //     FileWriter vectorWriter = new FileWriter("data/xor_data.csv");
    //     FileWriter labelWriter = new FileWriter("data/xor_labels.csv");

    //     for(int rounds = 0; rounds <= 10000; rounds++){
    //         int[] inputVector  = new int[2];
    //         for(int i = 0; i < 2; i++){
    //             inputVector[i] = (int)Math.round(Math.random());
    //         }

    //         int label = inputVector[0] ^ inputVector[1];

    //         vectorWriter.writeVectorInt(inputVector);
    //         labelWriter.writeLabel(label);
    //     }

    //     vectorWriter.close();
    //     labelWriter.close();
    // }

//     public static void main(String[] args) {
//         NeuralNetwork network = new NeuralNetwork.Builder(3, 2)
//         .addLayer(16, new ReLU())
//         .addLayer(2, new SoftMax())
//         .build();

//         FileParser dataParser = new FileParser("data/xor_data.csv", 2);
//         FileParser labelParser = new FileParser("data/xor_labels.csv", 1);
        

//         network.train(dataParser, labelParser, 1, 0, 1);

//     FileParser testData = new FileParser("data/xor_data.csv", 2);
    
//     FileParser testLabels = new FileParser("data/xor_labels.csv", 1);
//     int counter = 0;
//     while(testData.hasNextVector()){
//         double[] inputVector = testData.nextVector();
//         for(int i = 0; i < inputVector.length; i++){
//                    inputVector[i] = (inputVector[i]-0) / 1; // normalize input
//         }
//         double label = testLabels.nextDouble();
//         network.setInput(inputVector);
//         network.invoke();
//         double[] outputVector = network.getOuput();
//         int predictedLabel = Matrix.maxValueIndex(outputVector);
//         double val = outputVector[predictedLabel];
//         System.out.println("Vector no. " + counter +"predicted label: " + predictedLabel + "value:" + val);
//         if (predictedLabel == (int)label){
//             counter++;
//         }
// }
//     System.out.println("Accuracy: " + ((double)counter / 10000.0) * 100.0 + "%");
//     }
//  }