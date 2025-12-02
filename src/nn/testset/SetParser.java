package nn.testset;

import java.util.ArrayList;

import nn.file.FileParser;

public class SetParser {
    

    public static void parseTestSet(FileParser dataFile, FileParser labelFile,ArrayList<Vector> trainDataAndLabels) {

    while(dataFile.hasNextVector()){
        Vector vector = new Vector(dataFile.nextVector(), labelFile.nextDouble());
        trainDataAndLabels.add(vector);
    }
    
}
}
