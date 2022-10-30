import numpy as np

'''
Helper Functions to get the Weights of the Neural Networks
'''


def getNeuralNetworkWeights(layers, neuronInEachLayers, intializationType):
    weights = []
    for layer in range(layers-1):
        layer_weights = getWeights(
            neuronInEachLayers[layer+1], neuronInEachLayers[layer] + 1, intializationType)
        weights.append(layer_weights)
    return np.array(weights)


def getWeights(m, n, type):
    if type == 'zero':
        return np.zeros(shape=(m, n))
    if type == 'random':
        return 2*np.random.rand(m, n) - 1
    if type == 'normal':
        return np.random.randn(m, n)
    return -1


def getActivationValues(layers, neuronInEachLayers):
    a_values, z_values, δ_values = [], [], []
    for i in range(layers):
        a = np.ones(shape=(neuronInEachLayers[i] + 1, 1))
        z = np.ones(shape=(neuronInEachLayers[i] + 1, 1))
        δ = np.ones(shape=(neuronInEachLayers[i] + 1, 1))
        a_values.append(a)
        z_values.append(z)
        δ_values.append(δ)

    return np.array(a_values), np.array(z_values), np.array(δ_values)
