package naphon.ml.nn.errors;

import java.util.Arrays;

public interface ErrorFunction {
    double[] function(double[] predicted, double[] y);
    double[] derivative(double[] predicted, double[] y);
}
