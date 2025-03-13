#include <iostream>
#include <vector>
#include <string>
#include "DataProcessing.hpp"

int main() {
    std::vector<std::string> filenames = {"data1.csv", "data2.csv", "data3.csv"};
    std::vector<std::vector<double> > features;
    std::vector<double> target;
    readData(filenames, features, target);
    int populationSize = 100;
    int numFeatures = features.size();
    std::vector<std::vector<int> > population = initializePopulation(populationSize, numFeatures);
    double mutationRate = 0.01;
    double selectionRate = 0.5;
    int maxGenerations = 50;
    geneticAlgorithm(population, features, target, mutationRate, selectionRate, maxGenerations);
}
