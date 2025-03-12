#ifndef DATA_PROCESSING_HPP
#define DATA_PROCESSING_HPP
#include <vector>
#include <string>
#include <map>

// perform label encoding
void labelEncode(const std::vector<std::string>& column, std::map<std::string, int>& encoding);
// perform one-hot encoding
std::vector<std::vector<int> > oneHotEncode(const std::vector<std::string>& column);
// perform min-max scaling
std::vector<double> minMaxScale(const std::vector<double>& data);
// perform standardization
std::vector<double> standardize(const std::vector<double>& data);

#endif
