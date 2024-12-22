package naphon.ml;

import naphon.ml.nn.NeuralNetwork;
import naphon.ml.nn.errors.CrossEntropyLossFunction;
import naphon.ml.nn.layer.LinearLayer;
import naphon.ml.nn.layer.activation.ReLUActivationFunction;
import naphon.ml.nn.layer.activation.SigmoidActivationFunction;
import naphon.ml.nn.optimizer.AdaptiveMomentEstimationOptimizer;
import naphon.ml.nn.optimizer.Optimizer;
import naphon.ml.nn.optimizer.StochasticGradientDescentOptimizer;

import java.util.Arrays;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        DataFrame df = new DataFrame("x1","x2", "y");
        df.addRow(0,0,0);
        df.addRow(1,0,1);
        df.addRow(0,1,1);
        df.addRow(1,1,0);
        df.represent();
        Optimizer optim = new AdaptiveMomentEstimationOptimizer(.9, .99, 1e-8);

        NeuralNetwork nn = new NeuralNetwork(0.001, new CrossEntropyLossFunction(),
                new LinearLayer(2,100, optim),
                new SigmoidActivationFunction(),
                new LinearLayer(100,100, optim),
                new SigmoidActivationFunction(),
                new LinearLayer(100,20, optim),
                new SigmoidActivationFunction(),
                new LinearLayer(20,1, optim)
        );

        int totalEpoch = 1000;
        int sub = totalEpoch/10;
        for(int epoch = 0; epoch < totalEpoch ; epoch++){
            nn.train(df, new String[]{"x1","x2"}, new String[]{"y"});
            if(epoch % sub == 0)
                System.out.printf("Epoch: [%6d/%6d], Loss = %.6f\n", epoch, totalEpoch, nn.getLoss() >= 0 ? nn.getLoss() : -nn.getLoss());

        }
        double[] x = {1,0};
        double[] y = nn.forward(x);
        double loss = nn.getLoss() >= 0 ? nn.getLoss() : -nn.getLoss();;
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
