package nn.demo;
import nn.NeuralNetwork;
import nn.file.FileParser;
import nn.utils.ActivationFunctions.*;
class Main {
public static void main(String[] args){
    
    //FileParser parser = new FileParser("data/fashion_mnist_train_vectors.csv", 28*28);

    //int i = 0 ;
    //while(parser.hasNextVector()){
        //parser.nextVector();
        //System.out.println(i);
        //i++;
    //}

    NeuralNetwork network = new NeuralNetwork.Builder(3, 28*28)
        .addLayer(128, new ReLU())
        .addLayer(10, new SoftMax())
        .build();
    
    System.err.println(network);
    //for(int i = 0; i < newVector.length; i++){
     // ##  System.out.println(newVector[i]);
   // }

}}