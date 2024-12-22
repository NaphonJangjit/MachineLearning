package naphon.ml.nn.layer.activation;

import naphon.ml.nn.layer.Layer;

public interface ActivationFunction extends Layer {
    double activate(double x);
    double deactivate(double x);
}
