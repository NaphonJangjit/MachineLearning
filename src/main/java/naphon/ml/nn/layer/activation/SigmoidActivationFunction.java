package naphon.ml.nn.layer.activation;

public class SigmoidActivationFunction implements ActivationFunction {
    private double[] inputs;
    private double[] outputs;

    @Override
    public double[] forward(double[] input) {
        this.inputs = input;
        this.outputs = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            outputs[i] = activate(input[i]);
        }
        return outputs;
    }

    @Override
    public double[] backward(double[] outputError, double learningRate) {
        double[] inputError = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            inputError[i] = outputError[i] * deactivate(inputs[i]);
        }
        return inputError;
    }

    @Override
    public double activate(double x) {
        // Sigmoid function: 1 / (1 + e^(-x))
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double deactivate(double x) {
        // Derivative of sigmoid function: sigmoid(x) * (1 - sigmoid(x))
        double sigmoid = activate(x);
        return sigmoid * (1 - sigmoid);
    }
}
