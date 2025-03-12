#ifndef FEATURESELECTION_HPP
#define FEATURESELECTION_HPP
#include <vector>
#include <string>
#include <map>

// Initialize the population
std::vector<std::vector<bool>> initializePopulation(int populationSize, int numFeatures);
// Evaluate the fitness of each individual in the population
std::vector<double> evaluateFitness(const std::vector<std::vector<bool>> population, std::vector<double> fitnessValues);
//select parents based on fitness
std::vector<std::vector<bool>> selectParents(const std::vector<std::vector<double>>& features, const std::vector<int>& target, const std::vector<std::vector<bool>>& population);
// Perform crossover between two parents
std::vector<bool> crossover(const std::vector<std::vector<bool>>& parents);
// Perform mutation on an individual
void mutate(std::vector<std::vector<bool>>& population, double mutationRate);
// display results
void displayResults(const std::vector<std::vector<bool>>& population, const std::vector<double>& fitnessScores);
// Perform feature selection
std::vector<std::vector<bool>> featureSelection(const std::vector<std::vector<double>>& features, const std::vector<int>& target, int populationSize, int numGenerations, double mutationRate, double crossoverRate, double selectionRate);

#endif // FEATURESELECTION_HPP