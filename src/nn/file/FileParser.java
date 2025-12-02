package nn.file;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class FileParser {
    private Scanner scanner; 
    private int lineLength;
    private double normalizationSd;
    private double normalizationMean ;

    public FileParser(String filePath,double normalization_sd, double normalizationMean, int lineLength){
        this.normalizationSd = normalization_sd;
        this.normalizationMean = normalizationMean;
        File f = new File(filePath);
        try{
            this.scanner = new Scanner(f).useDelimiter(",|\\n"); // delimiter are both comma and newline 
        }
        catch(FileNotFoundException e){
            e.printStackTrace();
        }

        this.lineLength = lineLength;
    }

    public double nextDouble(){
        return scanner.nextDouble();
    }
    
    public double[] nextVector(){
        double[] vector = new double[lineLength];


        for(int position = 0; position < lineLength; position++){
            vector[position] = (scanner.nextDouble()- normalizationMean)/normalizationSd;
        }

        return vector;
    }

    public boolean hasNextVector(){

        return scanner.hasNext();
    }

    public void close(){
        scanner.close();
    }
}
