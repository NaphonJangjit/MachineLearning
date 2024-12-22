package naphon.ml.nn.layer;

public interface Layer {

    double[] forward(double[] input);
    double[] backward(double[] outputError, double learningRate);

}
