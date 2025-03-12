#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>
#include <iomanip>

// perform label encoding
void labelEncode(const std::vector<std::string>& column, std::map<std::string, int>& encodedLabels) {
    int index = 0;
    for (const auto& value : column) {
        if (encodedLabels.find(value) == encodedLabels.end()) {
            encodedLabels[value] = index++;
        }
    }
}

// perform one-hot encoding
std::vector<std::vector<int> > oneHotEncode(const std::vector<std::string>& column) {
    std::map<std::string, int> encoding;
    labelEncode(column, encoding);
    int numClasses = encoding.size();
    std::vector<std::vector<int> > oneHoEncode(column.size(), std::vector<int>(numClasses, 0));
    for (size_t i = 0; i < column.size(); ++i) {
        oneHoEncode[i][encoding[column[i]]] = 1;
    }
    return oneHoEncode;
}

// perform min-max scaling
std::vector<double> minMaxScale(const std::vector<double>& data) {
    double minVal = *min_element(data.begin(), data.end());
    double maxVal = *max_element(data.begin(), data.end());
    std::vector<double> scaledData;
    for (const auto& value : data) {
        scaledData.push_back((value - minVal) / (maxVal - minVal));
    }
    return scaledData;
}

// perform Standardisation
std::vector<double> standardize(const std::vector<double>& data) {
    double mean = 0.0;
    double stdDev = 0.0;
    std::vector<double> standardisedData;
    // Calculate mean
    for (const auto& value : data) {
        mean += value;
    }
    mean /= data.size();
    // Calculate standard deviation
    for (const auto& value : data) {
        stdDev += pow(value - mean, 2);
    }
    stdDev = sqrt(stdDev / data.size());
    // Standardise data
    for (const auto& value : data) {
        standardisedData.push_back((value - mean) / stdDev);
    }
    return standardisedData;
}