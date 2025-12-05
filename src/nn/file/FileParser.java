package nn.file;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

/**  Class for parsing the vector and label files */

public class FileParser {
    private Scanner scanner; 
    private int lineLength;
    private double normalizationSd;
    private double normalizationMean ;

    /**
     * 
     * @param filePath 
     * @param normalization_sd This double is used to to divide a parsed number
     * @param normalizationMean used to subtract from a parsed number
     * @param lineLength length of a line in the document
     */
    public FileParser(String filePath,double normalization_sd, double normalizationMean, int lineLength){
        this.normalizationSd = normalization_sd;
        this.normalizationMean = normalizationMean;
        File f = new File(filePath);
        try{
            this.scanner = new Scanner(f).useDelimiter(",|\\n"); 
        }
        catch(FileNotFoundException e){
            e.printStackTrace();
        }

        this.lineLength = lineLength;
    }

    public double nextDouble(){
        return scanner.nextDouble();
    }
    
    /**
     * 
     * @return next line of the file with size lineLength
     */
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
