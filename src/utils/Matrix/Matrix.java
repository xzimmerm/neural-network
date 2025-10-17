package utils.Matrix;

public final class Matrix {
    
    public static double arrayProductAndSum(double[] inputs, double[] weights){

        if (inputs.length + 1 != weights.length){
            throw new IllegalArgumentException("Arrays must be of the same length");
        }

        double result = weights[0]; // bias

        for(int i = 0; i < inputs.length; i++){
            result += inputs[i] * weights[i + 1];
        }
        return result;
        
    }
}
