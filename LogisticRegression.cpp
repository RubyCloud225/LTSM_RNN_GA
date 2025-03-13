#include<vector>
#include<cmath>
#include<iostream>

class LogisticRegression {
    public:
    LogisticRegression(double learningRate = 0.01, int iterations = 1000) : learningRate(learningRate), iterations(iterations) {}
    void fit(const std::vector<std::vector<double> >& X, const std::vector<double>& y) {
        int m = X.size(); // number of samples
        int n = X[0].size(); // number of features
        weights = std::vector<double>(n, 0.0); // initialize weights to 0
        
        for (int i = 0; i < iterations; ++i) {
            std::vector<double> predictions = predictProbabilities(X);
            for (int j = 0; j < n; ++j) {
                double gradient = 0.0;
                for (int k = 0; k < m; ++k) {
                    gradient += (predictions[k] - y[k]) * X[k][j];
                }
                weights[j] -= (learningRate / m) * gradient; // update weights
            }
        }
    }
    std::vector<double> predictProbabilities(const std::vector<std::vector<double> >& X) {
        std::vector<double> probabilities(X.size());
        for (int i = 0; i < X.size(); ++i) {
            probabilities[i] = sigmoid(dotProduct(X[i], weights));
        }
        return probabilities;
    }
    std::vector<int> predict(const std::vector<std::vector<double> >& X) {
        std::vector<double> probabilities = predictProbabilities(X);
        std::vector<int> predictions(probabilities.size());
        for (size_t i = 0; i < probabilities.size(); ++i) {
            predictions[i] = (probabilities[i] >= 0.5) ? 1 : 0; // Threshold ar 0.5
        }
        return predictions;
    }
    private:
    double learningRate;
    int iterations;
    std::vector<double> weights;
    double sigmoid(double z) {
        return 1 / (1 + std::exp(-z));
    }
    double dotProduct(const std::vector<double>& x, const std::vector<double>& w) {
        double result = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            result += x[i] * w[i];
        }
        return result;
    }
};