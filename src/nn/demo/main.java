package nn.demo;
import nn.file.FileParser;
class Main {
public static void main(String[] args){
    
    FileParser parser = new FileParser("data/fashion_mnist_train_vectors.csv", 28*28);

    int i = 0 ;
    while(parser.hasNextVector()){
        parser.nextVector();
        System.out.println(i);
        i++;
    }
    
    //for(int i = 0; i < newVector.length; i++){
     // ##  System.out.println(newVector[i]);
   // }

}}