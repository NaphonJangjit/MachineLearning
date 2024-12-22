package naphon.ml.nn.layer.activation;

public class ReLUActivationFunction implements ActivationFunction {

    private double[] inputs;
    private double[] outputs;

    @Override
    public double activate(double x) {
        return x >= 0 ? x : 0;
    }

    @Override
    public double deactivate(double x) {
        return x >= 0 ? 1 : 0;
    }

    @Override
    public double[] forward(double[] input) {
        this.inputs = input;
        this.outputs = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = activate(inputs[i]);
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
}
