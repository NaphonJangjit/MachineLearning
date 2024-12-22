package naphon.ml.nn.optimizer;

public interface Optimizer {

    void update(double[] weights, double[] weightGradients, double learningRate);
    void updateBias(double[] biases, double[] biasGradients, double learningRate);
}
