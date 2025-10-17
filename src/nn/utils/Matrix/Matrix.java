package nn.utils.Matrix;

public final class Matrix {
    
    public static double weightProductAndSum(double[] inputs, double[] weights){

        if (inputs.length + 1 != weights.length){
            throw new IllegalArgumentException("Arrays must be of the same-ish length");
        }

        double result = weights[0]; // bias

        for(int i = 0; i < inputs.length; i++){
            result += inputs[i] * weights[i + 1];
        }
        return result;
        
    }
}
