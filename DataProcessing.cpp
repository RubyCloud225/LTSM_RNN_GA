#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>
#include <iomanip>
#include <random>
#include "LogisticRegression.cpp"

// Read data from multiple CSV files
void readData(const std::vector<std::string>& filenames, std::vector<std::vector<double> >& features, std::vector<double>& target) {
    for (const auto& filename : filenames) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            // Read the header
            if (features.empty()) {
                std::string line;
                std::getline(file, line);
                std::istringstream iss(line);
                std::string header;
                while (std::getline(iss, header, ',')) {
                    features.push_back(std::vector<double>());
                }
                // Read the data
                while (getline(file, line)) {
                    std::istringstream iss(line);
                    std::string value;
                    std::vector<double> row;
                    while (std::getline(iss, value, ',')) {
                        row.push_back(std::stod(value));
                    }
                    target.push_back(static_cast<int>(row[row.size() - 1]));
                    row.pop_back();
                    features[row.size()].insert(features[row.size()].end(), row.begin(), row.end());
                }
                file.close();
            } else {
                std::cerr << "Error: Unable to open file " << filename << std::endl;
                return;
            }
        }
    }
}

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

//Function to initialize the population
std::vector<std::vector<int> > initializePopulation(int populationSize, int numFeatures) {
    std::vector<std::vector<int> > population(populationSize, std::vector<int>(numFeatures));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (auto& individual : population) {
        for (auto& gene : individual) {
            gene = (dis(gen) < 0.5);
        }
    }
    return population;
}

std::vector<double> evaluateFitness(const std::vector<std::vector<double> >& features, const std::vector<double>& target, const std::vector<std::vector<int> >& population) {
    std::vector<double> fitnessScores(population.size());

    for (size_t i = 0; i < population.size(); ++i) {
        // Extract selected features based on the individual's genes
        std::vector<std::vector<double> > selectedFeatures; // This should be a 2D vector

        for (size_t k = 0; k < features.size(); ++k) { // Iterate over each sample
            std::vector<double> selectedRow; // This will hold the features for the current sample
            for (size_t j = 0; j < population[i].size(); ++j) { // Iterate over each feature
                if (population[i][j]) { // If the gene is true, include the feature
                    selectedRow.push_back(features[k][j]); // Add the feature to the selected row
                }
            }
            if (!selectedRow.empty()) { // Only add if there are selected features
                selectedFeatures.push_back(selectedRow); // Add the selected row to the 2D vector
            }
        }

        // Check if selectedFeatures is empty
        if (selectedFeatures.empty()) {
            fitnessScores[i] = 0.0; // Assign a fitness score of 0 if no features are selected
            continue; // Skip to the next individual
        }

        // Train the logistic regression model
        LogisticRegression model;
        model.fit(selectedFeatures, target); // Pass the 2D vector to fit

        // Make predictions and calculate accuracy
        std::vector<int> predictions = model.predict(selectedFeatures);
        double correct = 0.0;
        for (size_t k = 0; k < predictions.size(); ++k) {
            if (predictions[k] == target[k]) {
                correct += 1.0;
            }
        }

        // Calculate accuracy as fitness score
        fitnessScores[i] = correct / predictions.size(); // Store accuracy as fitness score
    }

    return fitnessScores;
}

// Function to select parents based on fitness
std::vector<std::vector<int> > selectParents(const std::vector<std::vector<int> >& population, const std::vector<double>& fitnessScores) {
    std::vector<std::vector<int> > parents;
    std::vector<double> cumulativeFitness(fitnessScores.size());
    std::partial_sum(fitnessScores.begin(), fitnessScores.end(), cumulativeFitness.begin());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, cumulativeFitness.back());
    for (size_t i = 0; i < population.size() / 2; ++i) {
        double randomValue = dis(gen);
        auto it = std::lower_bound(cumulativeFitness.begin(), cumulativeFitness.end(), randomValue);
        size_t index = std::distance(cumulativeFitness.begin(), it);
        parents.push_back(population[index]);
    }
    return parents;
}

// perform crossover
std::vector<std::vector<int> > crossover(const std::vector<std::vector<int> >& parents) {
    std::vector<std::vector<int> > offspring;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, parents.size() - 1);
    for (size_t i = 0; i < parents.size(); i += 2) {
        std::vector<int> child1 = parents[i];
        std::vector<int> child2 = parents[i + 1];
        int crossoverPoint = dis(gen);
        for (size_t j = crossoverPoint; j < child1.size(); ++j) {
            std::swap(child1[j], child2[j]);
        }
        offspring.push_back(child1);
        offspring.push_back(child2);
    }
    return offspring;
}

// perform mutation
void mutate(std::vector<std::vector<int> >& population, double mutationRate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 1);
    for (auto& individual : population) {
        for (auto& gene : individual) {
            if (dis(gen) < mutationRate) {
                gene = !gene; // flip the gene
            }
        }
    }
}

// genetic algoritm function
void geneticAlgorithm(std::vector<std::vector<int> >& population, const std::vector<std::vector<double> >& features, const std::vector<double>& target, double mutationRate, double selectionRate, int maxGenerations) {
    for (int generation = 0; generation < maxGenerations; ++generation) {
        std::vector<double> fitnessScores = evaluateFitness(features, target, population);
        std::vector<std::vector<int> > parents = selectParents(population, fitnessScores);
        std::vector<std::vector<int> > offspring = crossover(parents);
        mutate(offspring, mutationRate);
        population = offspring;
        std::cout << "Generation: " << generation + 1 << std::endl;
    }
}
