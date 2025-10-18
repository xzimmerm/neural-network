package nn.file;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class FileParser {
    private Scanner scanner; 
    private int lineLength; 

    public FileParser(String filePath, int lineLength){
        File f = new File(filePath);
        try{
            this.scanner = new Scanner(f).useDelimiter(",|\\n"); // delimiter are both comma and newline 
        }
        catch(FileNotFoundException e){
            e.printStackTrace();
        }

        this.lineLength = lineLength;
    }

    public double[] nextVector(){
        double[] vector = new double[lineLength];


        for(int position = 0; position < lineLength; position++){
            vector[position] = scanner.nextDouble();
        }

        return vector;
    }

    public boolean hasNextVector(){

        if(scanner.hasNext()){
            return true; 
        }
        else{
            scanner.close();
            return false;
        }
    }
}
