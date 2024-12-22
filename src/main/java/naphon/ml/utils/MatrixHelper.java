package naphon.ml.utils;

public class MatrixHelper {

    public static double[][] transpose(double[][] d){
        double[][] out = new double[d[0].length][d.length];
        for(int i = 0; i < d.length; i++){
            for(int j = 0; j < d[0].length; j++){
                out[j][i] = d[i][j];
            }
        }

        return out;
    }

    public static void printMatrix(double[][] d){
        for(int i = 0; i < d.length; i++){
            for(int j = 0; j < d[0].length; j++){
                System.out.print(d[i][j] + " ");
            }
            System.out.println();
        }
    }

}
