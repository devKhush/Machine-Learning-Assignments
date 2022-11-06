import numpy as np


class ActivationFunction:
    def __init__(self, type) -> None:
        self.name = type

    @staticmethod
    def getActivation(type: str):
        function = {
            'sigmoid': Sigmoid(type),
            'tanh': TanH(type),
            'relu': ReLU(type)
        }
        return function[type]

    def function(self, z: np.ndarray) -> np.ndarray:
        pass

    def derivative(self, z: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(ActivationFunction):
    def function(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def derivative(self, z: np.ndarray) -> np.ndarray:
        y = self.function(z)
        return y * (1 - y)


class TanH(ActivationFunction):
    def function(self, z: np.ndarray) -> np.ndarray:
        return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        y = self.function(z)
        return 1 - np.square(y)


class ReLU(ActivationFunction):
    def function(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x: 1 if x >= 0 else 0)(z)


class SoftMax(ActivationFunction):
    def function(self, z: np.ndarray) -> np.ndarray:
        exponents = np.exp(z)
        return exponents / np.sum(exponents)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        y = self.function(z)
        return -np.outer(y, y) + np.diag(y.flatten())
