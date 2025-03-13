#ifndef DATAPROCESSING_HPP
#define DATAPROCESSING_HPP
#include <vector>
#include <string>
#include <map>

void readData(const std::vector<std::string>& filenames, std::vector<std::vector<double> >& features, std::vector<double>& target);
void labelEncode(const std::vector<std::string>& column, std::map<std::string, int>& encodedLabels);
std::vector<std::vector<int> > oneHotEncode(const std::vector<std::string>& column);
std::vector<double> minMaxScale(const std::vector<double>& data);
std::vector<double> standardize(const std::vector<double>& data);
std::vector<std::vector<int> > initializePopulation(int populationSize, int numFeatures);
std::vector<double> evaluateFitness(const std::vector<std::vector<double> >& features, const std::vector<double>& target, const std::vector<std::vector<int> >& population);
std::vector<std::vector<int> > selectParents(const std::vector<std::vector<int> >& population, const std::vector<double>& fitnessScores);
std::vector<std::vector<int> > crossover(const std::vector<std::vector<int> >& parents);
void mutate(std::vector<std::vector<int> >& population, double mutationRate);
void geneticAlgorithm(std::vector<std::vector<int> >& population, const std::vector<std::vector<double> >& features, const std::vector<double>& target, double mutationRate, double selectionRate, int maxGenerations);

#endif
