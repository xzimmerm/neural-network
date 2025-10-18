package nn.file;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class FileWriter {
    
    PrintWriter writer; 

    public FileWriter(String filePath){
        try{
            this.writer = new PrintWriter(new File(filePath));
        }
        catch(FileNotFoundException e){
            e.printStackTrace();
        }

    }

    public void writeVector(double[] vector){

        for(int position = 0; position < vector.length; position++){
            String component = Double.toString(vector[position]);
            if(position == vector.length - 1){
                
                component += '\n';

            }
            else {
                component += ',';
            }
            writer.write(component);
        }
    }

    public void writeLabel(int label){
        String labelString = Integer.toString(label) + '\n';
        writer.write(labelString);
    }

    public void close(){
        writer.close();
    }
}
