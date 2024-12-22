package naphon.ml.nn.errors;

public class CrossEntropyLossFunction implements ErrorFunction {

    @Override
    public double[] function(double[] predicted, double[] y) {
        double[] lossValues = new double[predicted.length];
        double[] errorVal = derivative(predicted, y);
        for(int i = 0; i < predicted.length; i++){
            lossValues[i] = errorVal[i] / (predicted[i]*(1-predicted[i]));
        }
        return lossValues;
    }

    @Override
    public double[] derivative(double[] predicted, double[] y) {
        double[] lossValues = new double[predicted.length];
        for(int i = 0; i < predicted.length; i++){
            lossValues[i] = (-(y[i] / predicted[i]) + ((1 - y[i]) / (1 - predicted[i])))*predicted[i]*(1-predicted[i]);
        }
        return lossValues;
    }
}
