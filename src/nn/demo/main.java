package nn.demo;
import nn.file.FileParser;
class Main {
public static void main(String[] args){
    
    FileParser parser = new FileParser("data/fashion_mnist_test_vectors.csv", 28*28);

    double[] newVector = parser.nextVector();
    System.out.println(newVector.length);
    
    //for(int i = 0; i < newVector.length; i++){
     // ##  System.out.println(newVector[i]);
   // }

    newVector = parser.nextVector();
     for(int i = 0; i < newVector.length; i++){
        System.out.println(newVector[i]);
    }
}}