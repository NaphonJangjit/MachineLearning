package naphon.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;

public class DataFrame {

    private LinkedHashMap<String, ArrayList<Double>> data;

    // Constructor to initialize an empty DataFrame
    public DataFrame(){
        data = new LinkedHashMap<>();
    }

    // Constructor to initialize DataFrame with specified column names (empty columns)
    public DataFrame(String...cols){
        data = new LinkedHashMap<>();
        for(String c : cols){
            data.put(c, new ArrayList<>());
        }
    }

    // Constructor to initialize DataFrame with provided data (LinkedHashMap)
    public DataFrame(LinkedHashMap<String, ArrayList<Double>> data){
        this.data = data;
    }

    // Gets the value at a specific column and row index
    public double get(String colName, int idx){
        if(!data.containsKey(colName) || data.get(colName).size() <= idx) {
            return -1;  // Return -1 if column does not exist or index is invalid
        }
        return data.get(colName).get(idx);
    }

    // Gets the value at a specific column where idxValue is found in idxCol
    public double get(String colName, String idxCol, double idxValue){
        if(!data.containsKey(idxCol) || !data.containsKey(colName)) {
            return -1;  // Return -1 if the columns don't exist
        }
        int idx = data.get(idxCol).indexOf(idxValue);
        if (idx == -1) {
            return -1;  // Return -1 if idxValue is not found
        }
        return data.get(colName).get(idx);
    }

    // Returns a column's data as a List
    public List<Double> getColumn(String colName){
        if(!data.containsKey(colName)){
            throw new IllegalArgumentException("Column " + colName + " does not exist!");
        }
        return data.get(colName);
    }

    // Returns a 2D array of data from multiple columns
    public double[][] getColumns(String[] colNames){
        int rows = data.get(colNames[0]).size();
        double[][] dat = new double[colNames.length][rows];
        for (int i = 0; i < colNames.length; i++) {
            List<Double> column = getColumn(colNames[i]);
            for (int j = 0; j < rows; j++) {
                dat[i][j] = column.get(j);
            }
        }
        return dat;
    }

    // Returns a 2D array of a single column in a 1xN shape
    public double[][] getSingleColumn(String colName){
        List<Double> column = getColumn(colName);
        double[][] dat = new double[1][column.size()];
        for (int i = 0; i < column.size(); i++) {
            dat[0][i] = column.get(i);
        }
        return dat;
    }

    // Sets the value in a specific column and row index
    public void set(String colName, int idx, double value){
        if(!data.containsKey(colName) || data.get(colName).size() <= idx) {
            return;  // If the column doesn't exist or index is invalid, do nothing
        }
        data.get(colName).set(idx, value);
    }

    // Sets the value in colName at the row where idxValue is found in idxCol
    public void set(String colName, String idxCol, double idxValue, double value){
        if(!data.containsKey(idxCol) || !data.containsKey(colName)) {
            return;  // If the columns don't exist, do nothing
        }
        int idx = data.get(idxCol).indexOf(idxValue);
        if (idx == -1) {
            return;  // If idxValue is not found, do nothing
        }
        data.get(colName).set(idx, value);
    }

    // Removes a row at a specific index
    public void removeRow(int idx){
        if (idx < 0 || idx >= data.values().iterator().next().size()) {
            return;  // If index is out of bounds, do nothing
        }
        for (String key : data.keySet()) {
            data.get(key).remove(idx);
        }
    }

    // Removes a row where idxValue is found in idxCol
    public void removeRow(String idxCol, double idxValue){
        if(!data.containsKey(idxCol)) {
            return;  // If idxCol does not exist, do nothing
        }
        int idx = data.get(idxCol).indexOf(idxValue);
        if (idx == -1) {
            return;  // If idxValue is not found, do nothing
        }
        for (String key : data.keySet()) {
            data.get(key).remove(idx);
        }
    }

    // Prints the DataFrame in a tabular format
    public void represent() {
        // Check if the data is empty
        if (data.isEmpty()) {
            System.out.println("The DataFrame is empty.");
            return;
        }

        // Print the header (column names)
        System.out.print(" | ");
        for (String colName : data.keySet()) {
            System.out.print(colName + " | ");
        }
        System.out.println();

        // Get the number of rows (assuming all columns have the same number of elements)
        int rowCount = data.values().iterator().next().size();

        // Print the rows
        for (int i = 0; i < rowCount; i++) {
            System.out.print(" | ");
            for (String colName : data.keySet()) {
                // Ensure to print each value in the column
                ArrayList<Double> columnData = data.get(colName);
                if (i < columnData.size()) {
                    System.out.print(columnData.get(i) + " | ");
                } else {
                    System.out.print("null | ");  // Handle missing values
                }
            }
            System.out.println();
        }
    }

    // Adds a row of data to the DataFrame
    public void addRow(double...d){
        int i = 0;
        if (d.length != data.size()) {
            System.err.println("Data length does not match the number of columns.");
            return;  // If the data doesn't match the column size, do nothing
        }
        for (String key : data.keySet()) {
            data.get(key).add(d[i]);
            i++;
        }
    }
}
