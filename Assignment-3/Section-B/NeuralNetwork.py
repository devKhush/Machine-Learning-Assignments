import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ActivationFunction import *
from WeightInitialization import *

'''
Artifical Neural Network (or Multilayer Perceptron)

Author: Khushdev Pandit <khushdev20211@iiitd.ac.in, khushdev7838@gmail.com>
'''


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
        >>> batchSize = Batch size to be used in Mini-Batch Gradient Descent
        '''
        self.L = N + 2
        self.neuronInEachLayers = neuronInEachLayers
        self.lr = lr
        self.epochs = epochs
        self.batchSize = batchSize

        # Activation function to be used
        self.activation: ActivationFunction = ActivationFunction.getActivation(
            activation)
        # Model weights
        self.weights_Θ: np.ndarray = getNeuralNetworkWeights(
            self.L, neuronInEachLayers, weightInitType)
        # Values of Activation (a = g(z)), Weighted Sum (z), Delta/Error terms (δ)
        self.a_values, self.z_values, self.δ_values = getActivationValues(
            self.L, neuronInEachLayers)
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
        self.a_values[L -
                      1][1:] = SoftMax('softmax').function(self.z_values[L-1][1:])
        return self.a_values[L-1][1:].copy()

    def backwardPropagation(self, y: np.ndarray) -> np.ndarray:
        '''
        Calculates the error terms (δ) of all the neurons in each layers using BackProgagation Algorithm
        '''
        # Backward Progagation on Output layer using the Softmax Activation Function's Derivative
        y = np.array([y]).T
        self.δ_values[self.L-1][1:] = np.dot(SoftMax('softmax').derivative(
            self.z_values[self.L-1][1:]), y - self.a_values[self.L-1][1:])
        # self.δ_values[self.L-1][1:] = y - self.a_values[self.L-1][1:]

        # Backward Progagation on Hidden layer using the actual/input Activation Function
        for l in range(self.L-2, 0, -1):
            self.δ_values[l] = np.dot(self.weights_Θ[l].T, self.δ_values[l+1][1:]) * \
                self.activation.derivative(self.z_values[l])
        return

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series):
        '''
        Fit the Model Parameters into the Neural Network
        '''
        # Array to store Training and Validation cross-entropy loss 
        train_loss, validation_loss = [0], [0]

        # Run over all the epochs
        for epoch in range(self.epochs):
            print(epoch, end=' ')

            # In each epoch, train the Neural network model Batch wise
            for batch_index in range(0, x_train.shape[0], self.batchSize):
                # Change in wieghts captured on the whole training Batch
                Δ_weights = getNeuralNetworkWeights(
                    self.L, self.neuronInEachLayers, 'zero')

                # Propagate each Training example in the Training Batch forward and backward
                # Also, calculate the error terms (δ) for all the layers 
                for i in range(batch_index, min(batch_index+self.batchSize, x_train.shape[0]-1)):
                    self.forwardPropagation(x_train.iloc[i])
                    self.backwardPropagation(y_train[i])

                    for l in range(self.L-2, -1, -1):
                        Δ_weights[l] = Δ_weights[l] + \
                            np.dot(self.δ_values[l+1][1:], self.a_values[l].T)

                # Update the weights Batch wise
                self.weights_Θ = self.weights_Θ + self.lr * Δ_weights

            # Calculate the Log-loss (or Cross-Entropy loss) on Training and validation 
            train_loss.append(self.cross_entropy_loss(x_train, y_train))
            validation_loss.append(self.cross_entropy_loss(x_test, y_test))

        plt.figure(figsize=(8, 8))
        plt.plot(train_loss, 'r-o', label='Training Loss', ms=4)
        plt.plot(validation_loss, 'b-o', label='Validation Loss', ms=4)
        plt.xlabel('Iterations', fontsize=13)
        plt.ylabel('Loss/Error', fontsize=13)
        plt.xlim((0.75, self.epochs))
        plt.title(
            f"Training loss & Validation loss v/s Iterations for Activation='{self.activation.name}'\nPlot of Cross Entropy Loss", fontsize=15)
        plt.legend()
        plt.show()
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
        return np.array(y_pred).copy()

    def score(self, x_test: pd.DataFrame, y_test: pd.Series):
        ''' 
        Calculate the accuracy score of the trained model in input "x_test" and labels "y_test" 
        '''
        y_pred = self.predict(x_test)
        y_test = np.argmax(y_test, axis=1)
        return (y_test == y_pred[:, 0]).sum()/y_test.shape[0]

    def cross_entropy_loss(self, x_test: pd.DataFrame, y_test: np.ndarray):
        '''
        Calculate the Cross Entropy loss on the given Dataset.
        This is same as sklearn's log_loss
        '''
        y_pred_proba = self.predict_proba(x_test).reshape(y_test.shape[0], 10)
        cross_entrpy_loss = y_test * np.log2(y_pred_proba)
        return -np.sum(cross_entrpy_loss) / x_test.shape[0]
