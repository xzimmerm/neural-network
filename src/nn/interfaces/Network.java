package nn.interfaces;

public interface Network {

    public void train();

    public void invoke();

    public void setInput(double[] inputVector);

    public double[] getOuput();
}