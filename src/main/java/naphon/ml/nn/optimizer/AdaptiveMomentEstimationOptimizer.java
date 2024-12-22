package naphon.ml.nn.optimizer;

import java.util.HashMap;
import java.util.Map;

public class AdaptiveMomentEstimationOptimizer implements Optimizer {
    private double beta1;
    private double beta2;
    private double epsilon;
    private Map<Object, double[]> m;
    private Map<Object, double[]> v;
    private int t;

    public AdaptiveMomentEstimationOptimizer(double beta1, double beta2, double epsilon) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.m = new HashMap<>();
        this.v = new HashMap<>();
        this.t = 0;
    }

    private double[] initializeIfAbsent(Object key, Map<Object, double[]> map, int size) {
        return map.computeIfAbsent(key, k -> new double[size]);
    }

    @Override
    public void update(double[] weights, double[] weightGradients, double learningRate) {
        t++;
        double[] m_t = initializeIfAbsent(weights, m, weights.length);
        double[] v_t = initializeIfAbsent(weights, v, weights.length);

        for (int i = 0; i < weights.length; i++) {
            m_t[i] = beta1 * m_t[i] + (1 - beta1) * weightGradients[i];
            v_t[i] = beta2 * v_t[i] + (1 - beta2) * (weightGradients[i] * weightGradients[i]);

            double m_t_hat = m_t[i] / (1 - Math.pow(beta1, t));
            double v_t_hat = v_t[i] / (1 - Math.pow(beta2, t));

            weights[i] -= learningRate * m_t_hat / (Math.sqrt(v_t_hat) + epsilon);
        }
    }

    @Override
    public void updateBias(double[] biases, double[] biasGradients, double learningRate) {
        t++;
        double[] m_t = initializeIfAbsent(biases, m, biases.length);
        double[] v_t = initializeIfAbsent(biases, v, biases.length);

        for (int i = 0; i < biases.length; i++) {
            m_t[i] = beta1 * m_t[i] + (1 - beta1) * biasGradients[i];
            v_t[i] = beta2 * v_t[i] + (1 - beta2) * (biasGradients[i] * biasGradients[i]);

            double m_t_hat = m_t[i] / (1 - Math.pow(beta1, t));
            double v_t_hat = v_t[i] / (1 - Math.pow(beta2, t));

            biases[i] -= learningRate * m_t_hat / (Math.sqrt(v_t_hat) + epsilon);
        }
    }
}
