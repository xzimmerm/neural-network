package nn.testset;

/**
 * Class used to hold the input vector and its label together
 */
public class Vector {
    
    public double[] data;
    public double label;

    /**
     * 
     * @param data input vector
     * @param label its label
     */
    public Vector(double[] data, double label){
        this.data = data;
        this.label = label;
    }
}
