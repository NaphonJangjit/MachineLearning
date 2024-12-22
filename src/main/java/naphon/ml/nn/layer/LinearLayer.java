package naphon.ml.nn.layer;

import naphon.ml.nn.optimizer.Optimizer;

import java.util.Random;

public class LinearLayer implements Layer {
    private int inputSize, outputSize;
    private double[][] weights;
    private double[] biases, inputs, outputs;
    private Optimizer optimizer;

    public LinearLayer(int inputSize, int outputSize, Optimizer optimizer) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.optimizer = optimizer;
        initializeWeightsAndBiases();
    }

    private void initializeWeightsAndBiases() {
        Random random = new Random();
        weights = new double[outputSize][inputSize];
        biases = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = random.nextGaussian() * Math.sqrt(2.0 / inputSize);
            }
            biases[i] = random.nextGaussian() * 0.01;
        }
    }

    @Override
    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size mismatch.");
        }
        this.inputs = input;
        this.outputs = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputs[i] = biases[i];
            for (int j = 0; j < inputSize; j++) {
                outputs[i] += weights[i][j] * input[j];
            }
        }
        return outputs;
    }

    @Override
    public double[] backward(double[] outputError, double learningRate) {
        // Initialize arrays to store gradients and propagate input error
        double[] inputError = new double[inputSize];
        double[][] weightGradients = new double[outputSize][inputSize];
        double[] biasGradients = new double[outputSize];

        // Compute gradients of weights, biases, and input errors
        for (int i = 0; i < outputSize; i++) {
            biasGradients[i] = outputError[i]; // Bias gradient is simply the output error for each neuron
            for (int j = 0; j < inputSize; j++) {
                // Weight gradient is the input multiplied by the output error
                weightGradients[i][j] = outputError[i] * inputs[j];
                // Propagate error to the inputs
                inputError[j] += weights[i][j] * outputError[i];
            }
        }

        // Update weights and biases using the optimizer
        for (int i = 0; i < outputSize; i++) {
            optimizer.update(weights[i], weightGradients[i], learningRate);
        }
        optimizer.updateBias(biases, biasGradients, learningRate);

        return inputError; // Return input errors to propagate back further in the network
    }

}

