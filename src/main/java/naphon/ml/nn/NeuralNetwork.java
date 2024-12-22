package naphon.ml.nn;

import naphon.ml.DataFrame;
import naphon.ml.nn.errors.ErrorFunction;
import naphon.ml.nn.layer.Layer;
import naphon.ml.utils.MatrixHelper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class NeuralNetwork{

    private final ArrayList<Layer> layers = new ArrayList<>();
    private double learningRate;
    private double loss = 0;
    private ErrorFunction lossFunction;
    public NeuralNetwork(double learningRate,ErrorFunction lossFunction, Layer...layers){
        this.learningRate = learningRate;
        this.lossFunction = lossFunction;
        Collections.addAll(this.layers, layers);
    }

    private final void addLayer(Layer layer){
        layers.add(layer);
    }

    public void train(DataFrame df, String[] inputColumnName, String outputColumnName){
        double[][] inputs = df.getColumns(inputColumnName);
        double[][] targets = df.getSingleColumn(outputColumnName);
        double[][] tinputs = MatrixHelper.transpose(inputs);
        double[][] ttargets = MatrixHelper.transpose(targets);
        for(int row = 0; row < tinputs.length; row++){
            double[] inputRow = tinputs[row];
            double[] targetRow = ttargets[row];

            double[] outputs = forward(inputRow);

            double[] errors = lossFunction.derivative(outputs, targetRow);
            loss = Arrays.stream(errors).sum();
            loss /= errors.length + 0.0;
            for(int i = layers.size() - 1; i >= 0; i--){
                errors = layers.get(i).backward(errors, learningRate);
            }
        }
    }

    public double[] forward(double[] input){
        double[] currentInput = input;
        for(Layer layer : layers){
            currentInput = layer.forward(currentInput);
        }

        return currentInput;
    }

    public double getLoss() {
        return loss;
    }
}
