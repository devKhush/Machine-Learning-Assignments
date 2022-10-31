import numpy as np
import pandas as pd
from ActivationFunction import *
from WeightInitialization import *


class NeuralNetwork:
    '''
    Custom class for Nueral Network
    '''

    def __init__(self, N, neuronInEachLayers, lr, activation, weightInitType, epochs, batchSize) -> None:
        '''
        >>> L = numbers of layers
        >>> neuronInEachLayers = number of neurons in each layers
        >>> lr = learning rate
        >>> activation = Activation to be used on Hidden layers.
            Must be ['sigmoid', 'ReLU', 'TanH']
        >>> weightInitType = How to initialize weights.
            Must be ['zero', 'random', 'normal]
        >>> epochs = Epochs to be used in Gradient Descent
        >>> batchSize = ???
        '''
        self.L = N
        self.neuronInEachLayers = neuronInEachLayers
        self.lr = lr
        self.epochs = epochs
        self.batchSize = batchSize

        # Activation function to be used
        self.activation: ActivationFunction = ActivationFunction.getActivation(
            activation)
        # Model weights
        self.weights_Θ: np.ndarray = getNeuralNetworkWeights(
            N, neuronInEachLayers, weightInitType)
        # Values of Activation (a = g(z)), Weighted Sum (z), Delta/Error terms (δ)
        self.a_values, self.z_values, self.δ_values = getActivationValues(
            N, neuronInEachLayers)
        return

    def forwardPropagation(self, x: np.ndarray) -> np.ndarray:
        '''
        Predict the output for the input 'x', by performing the forward propagartion on input x
        '''
        self.a_values[0][1:] = np.array([x]).T

        # Forward Progagation on Hidden layers using the actual/input Activation Function
        for l in range(self.L-2):
            self.z_values[l+1][1:] = np.dot(self.weights_Θ[l],
                                            self.a_values[l])
            self.a_values[l+1][1:] = \
                self.activation.function(self.z_values[l+1][1:])

        # Forward Progagation on Output layer using the Softmax Activation Function
        L = self.L
        self.z_values[L-1][1:] = np.dot(
            self.weights_Θ[L-2], self.a_values[L-2])
        self.a_values[L-1][1:] = SoftMax().function(self.z_values[L-1][1:])
        return self.a_values[L-1][1:].copy()

    def backwardPropagation(self, y: np.ndarray) -> np.ndarray:
        '''
        Calculates the error terms (δ) of all the neurons in each layers using BackProgagation Algorithm
        '''
        # self.δ_values[self.L-1][1:] = np.dot(SoftMax().derivative(
        #     self.z_values[self.L-1][1:]), y - self.a_values[self.L-1][1:])
        self.δ_values[self.L-1][1:] = y - self.a_values[self.L-1][1:]

        for l in range(self.L-2, 0, -1):
            self.δ_values[l] = np.dot(self.weights_Θ[l].T, self.δ_values[l+1][1:]) * \
                self.activation.derivative(self.z_values[l])
        return

    # TODO: Remove???
    def resetGradients(self):
        '''
        Reset the Weights
        '''
        for l in range(self.L-2, -1, -1):
            self.weights_Θ[l] += self.lr * \
                np.dot(self.δ_values[l+1][1:], self.a_values[l].T)
        return

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        '''
        Fit the Model Parameters into the Neural Network
        '''
        # for epoch in range(self.epochs):
        #     print(epoch, end=' ')
        #     Δ_weights = getNeuralNetworkWeights(
        #         self.L, self.neuronInEachLayers, 'zero')

        #     for i in range(x_train.shape[0]):
        #         self.forwardPropagation(x_train.iloc[i])
        #         self.backwardPropagation(y_train[i])

        #         for l in range(self.L-2, -1, -1):
        #             Δ_weights[l] = Δ_weights[l] + \
        #                 np.dot(self.δ_values[l+1][1:], self.a_values[l].T)

        #     Δ_weights = Δ_weights / x_train.shape[0]
        #     self.weights_Θ = self.weights_Θ + self.lr * Δ_weights

        for epoch in range(self.epochs):
            print(epoch, end=' ')

            for i in range(x_train.shape[0]):
                self.forwardPropagation(x_train.iloc[i])
                self.backwardPropagation(y_train[i])

                Δ_weights = getNeuralNetworkWeights(
                    self.L, self.neuronInEachLayers, 'zero')
                for l in range(self.L-2, -1, -1):
                    Δ_weights[l] = Δ_weights[l] + \
                        np.dot(self.δ_values[l+1][1:], self.a_values[l].T)

                self.weights_Θ = self.weights_Θ + self.lr * Δ_weights
        return

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        '''
        Predict the Output class for an input 'x', by performing the forward propagartion on input x.
        '''
        y_pred = self.predict_proba(x_test)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, x_test: pd.DataFrame) -> np.ndarray:
        '''
        Predict the Probabilities for the all the Output classes/categories for input 'x_test',
        by performing the forward propagartion on input x.
        '''
        y_pred = []
        for i in range(x_test.shape[0]):
            y_pred.append(self.forwardPropagation(x_test.iloc[i]))
        return np.array(y_pred)

    def score(self, x_test: pd.DataFrame, y_test: pd.Series):
        ''' 
        Calculate the accuracy score of the trained model in input x_test and labels y_test 
        '''
        y_pred = self.predict(x_test)
        y_test = np.argmax(y_test, axis=1)
        return (y_test[:, 0] == y_pred[:, 0]).sum()/y_test.shape[0]
