** LSTM-RNN Classifier with GA Arcitecture 
* https://www.mdpi.com/2078-2489/11/5/243

- data preprocessing - numericalization 38 numeric feautues and three non numeric features. Non Numeric needs to be converted to numerical- encoded and one hot encoded. transform to binary

- normalisation - standardisation - min max scaling - normalisation - standardisation - min max scaling

- Feature selection 
initalize the population - fitness value assignment - crossover - mutation - selection - replacement

- GA parameters - population size - number of generations - crossover probability - mutation probability - selection method -

- initialize the population - mapped 10 feature sets randomly selected from 122 attributes. 
    - each attribute is a feature - each feature is a binary value - 0 or 1 - 0 means not included in the feature
- Fitness Value Assignment -
    - Assign a fitness value of the individual feature set - use a logistic regression model to predict the target variable - use the accuracy of the model as the fitness value - the lowest selection error represents highest fitness

- Crossover -
    - select two parent feature sets - select a random crossover point - create two child feature sets - combine through a uniform crossover method

- Mutation -
    - flips the value of attribute to input individual and returns new individual. 
    - the probablity of mutation is 0.1 increase the search ability of the algorithm

- Selection -
    select the individual with highest predictive feature set selection and then terminate. 
    continue the process until the termination criteria is met

- Replacement -
    - replace the least fit individual with the new individual
    - repeat the process until the termination criteria is met

- LSTM-RNN
    - Batch size - 32
    - Number of epochs - 100
    - Optimizer - Adam
    - Loss function - binary cross entropy
    - Activation function - sigmoid
    - Number of hidden layers - 2
    - Number of neurons in each hidden layer - 128
    - Dropout - 0.2
    - Bidirectional - True
    - LSTM - True
    - RNN - True
    - Dense - True
    - Dense units - 128
    - Dense activation - sigmoid
    - Dense dropout - 0.2
    - Output activation - sigmoid
    - Output dropout - 0.2
    - Output shape - (None, 1)
    - Input shape - (None, 10, 1)
    - Input dimension - 1
    - Input length - 10
    - Input features - 1
    - Input type - float32
    - Input mode - sequential
    - Input padding - 'pre'
    - Input truncation - None

