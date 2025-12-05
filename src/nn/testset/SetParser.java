package nn.testset;

import java.util.ArrayList;

import nn.file.FileParser;

/**
 * Used to parse a vector and a label file into an ArrayList of Vectors
 */
public class SetParser {
    
    /**
     * 
     * @param dataFile vector file
     * @param labelFile label file
     * @param trainDataAndLabels the out parameter
     */
    public static void parseTestSet(FileParser dataFile, FileParser labelFile,ArrayList<Vector> trainDataAndLabels) {

    while(dataFile.hasNextVector()){
        Vector vector = new Vector(dataFile.nextVector(), labelFile.nextDouble());
        trainDataAndLabels.add(vector);
    }
    
}
}
