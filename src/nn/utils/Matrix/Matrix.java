package nn.utils.Matrix;

/**
 * Class for util operations with arrays and matrixes
 */
public final class Matrix {
    
    /**
     * used for neuron potentials
     * @param inputs the outputs of neurons from the previous layer
     * @param weights neuron weights
     * @return neuron potential
     */
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

    /**
     * 
     * @param addingTo an out parameter, matrix that is being added to
     * @param zeroing matrix that is zeroed out
     * @param batchSize used to divide the sum
     * @param learningRate
     */
    public static void addAndZeroOutMatrix(double[][][] addingTo, double[][][] zeroing, int batchSize,double learningRate){
        for(int layer = 1; layer < addingTo.length; layer++){
            for(int neuron = 0; neuron < addingTo[layer].length; neuron++){
                for(int weight = 0; weight < addingTo[layer][neuron].length; weight++){
                   
                    addingTo[layer][neuron][weight] += (zeroing[layer][neuron][weight] / batchSize)*(-learningRate);
                    zeroing[layer][neuron][weight] = 0;
                }
            }
        }
    }

    /**
     * 
     * @param vector
     * @return: index of the max value inside an array
     */
    public static int maxValueIndex(double[] vector){
        int maxIndex = 0;
        double maxValue = vector[0];

        for(int i = 1; i < vector.length; i++){
            if(vector[i] > maxValue){
                maxValue = vector[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}
