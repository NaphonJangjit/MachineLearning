package naphon.ml;

import naphon.ml.nn.NeuralNetwork;
import naphon.ml.nn.errors.CrossEntropyLossFunction;
import naphon.ml.nn.layer.LinearLayer;
import naphon.ml.nn.layer.activation.ReLUActivationFunction;
import naphon.ml.nn.optimizer.Optimizer;
import naphon.ml.nn.optimizer.StochasticGradientDescentOptimizer;

import java.util.Arrays;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        DataFrame df = new DataFrame("x1","x2", "y");
        //xor problem
        for(int i = 0; i < 100; i++) df.addRow(createFunction());
        df.represent();
        Optimizer optim = new StochasticGradientDescentOptimizer();
        NeuralNetwork nn = new NeuralNetwork(0.0001, new CrossEntropyLossFunction(), new LinearLayer(2,10, optim), new ReLUActivationFunction(), new LinearLayer(10,10, optim), new ReLUActivationFunction(), new LinearLayer(10,1, optim));


        for(int epoch = 0; epoch < 1000 ; epoch++){
            System.out.print(epoch % 10 == 0 ? "Epoch: " + epoch + "\n" : "");
            nn.train(df, new String[]{"x1","x2"}, "y");
        }
        double[] x = {1,2};
        double[] y = nn.forward(x);
        double loss = nn.getLoss();
        System.out.printf("Input: %s, Output: %f, loss = %f\n", Arrays.toString(x), y[0], loss);

    }

    public static double[] createFunction(){
        Random random = new Random();
        double x1 = random.nextDouble();
        double x2 = random.nextDouble();
        double answer = 2*x1 + 3*x2;
        return new double[]{x1,x2,answer};
    }

}
