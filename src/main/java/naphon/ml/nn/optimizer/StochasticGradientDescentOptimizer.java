package naphon.ml.nn.optimizer;

public class StochasticGradientDescentOptimizer implements Optimizer {
    @Override
    public void update(double[] weights, double[] weightGradients, double learningRate) {
        for(int i = 0; i < weights.length; i++){
            weights[i] -= learningRate * weightGradients[i];
        }
    }

    @Override
    public void updateBias(double[] biases, double[] biasGradients, double learningRate) {
        for(int i = 0; i < biases.length; i++){
            biases[i] -= learningRate * biasGradients[i];
        }
    }
}
