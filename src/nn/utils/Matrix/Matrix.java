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

    public static void zeroOutMatrix(double[][][] matrix){
        for(int layer = 1; layer < matrix.length; layer++){
            for(int neuron = 0; neuron < matrix[layer].length; neuron++){
                for(int weight = 0; weight < matrix[layer][neuron].length; weight++){
                    matrix[layer][neuron][weight] = 0;
                }
            }
        }
    }

    public static void addAndZeroOutMatrix(double[][][] addingTo, double[][][] zeroing){
        for(int layer = 1; layer < addingTo.length; layer++){
            for(int neuron = 0; neuron < addingTo[layer].length; neuron++){
                for(int weight = 0; weight < addingTo[layer][neuron].length; weight++){
                   
                    addingTo[layer][neuron][weight] += zeroing[layer][neuron][weight];
                    zeroing[layer][neuron][weight] = 0;
                }
            }
        }
    }
}
