#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include "DataProcessing.hpp"

using namespace std;

int main() {
    ifstream file("data.csv");
    string line;
    vector<vector<string> > data;
    vector<string> headers;

    // read the csv file
    if (file.is_open()) {
        getline(file, line);
        stringstream ss(line);
        string header;
        while (getline(ss, header, ',')) {
            headers.push_back(header);
        }
        while (getline(file, line)) {
            stringstream ss(line);
            string value;
            vector<string> row;
            while (getline(ss, value, ',')) {
                row.push_back(value);
            }
            data.push_back(row);
        }
        file.close();
    }
    // process non numerical data (assumed first column is categorical)
    vector<string> categoricalColumn;
    for (const auto& row : data) {
        categoricalColumn.push_back(row[0]);
    }
    // one-hot encode the categorical data
    vector<vector<int> > oneHotEncodedData = oneHotEncode(categoricalColumn);

    // process numerical data
    vector<double> numericalColumn;
    for (const auto& row : data) {
        numericalColumn.push_back(stod(row[1]));
    }
    // Standardize numerical data
    vector<double> standardisedData = standardize(numericalColumn);
    // Min -Max scale numerical data
    vector<double> minMaxScaledData = minMaxScale(standardisedData);
    // output results
    cout << "One-hot encoded data:" << endl;
    for (const auto& row : oneHotEncodedData) {
        for (const auto& value : row) {
            cout << value << " ";
        }
        cout << endl;
    }
    cout << "Standardized data:" << endl;
    for (const auto& value : standardisedData) {
        cout << fixed << setprecision(4) << value << " ";
    }
    cout << endl;
    cout << "Min-Max scaled data:" << endl;
    for (const auto& value : minMaxScaledData) {
        cout << fixed << setprecision(4) << value << " ";
    }
    cout << endl;
    return 0;            
}