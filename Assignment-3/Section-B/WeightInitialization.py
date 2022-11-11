import numpy as np

'''
Helper Functions to get the Weights of the Neural Networks

Author: Khushdev Pandit <khushdev20211@iiitd.ac.in, khushdev7838@gmail.com>
'''


def getNeuralNetworkWeights(layers, neuronInEachLayers, intializationType):
    '''
    Return the weights of Neural networks basd on the layers size and weight type
    layers -> no. of layers
    neuronInEachLayers -> neurons in each layers
    intializationType -> type of weights
    '''
    # Weight of neural networks
    weights = []

    # Get weights of each layers one by one
    for layer in range(layers-1):
        layer_weights = getWeights(
            neuronInEachLayers[layer+1], neuronInEachLayers[layer] + 1, intializationType)
        weights.append(layer_weights)

    return np.array(weights)


def getWeights(m, n, type):
    '''
    Return the weight of a particular layer based on the no. of neuron as inputs and outputs
    m -> no. of outputs from that layers
    n -> no. of inputs into that layers
    type -> type of weights
    '''
    if type == 'zero':
        return np.zeros(shape=(m, n)) / np.sqrt(m+n)
    if type == 'random':
        return 0.3*np.random.rand(m, n) / np.sqrt(m+n)
    if type == 'normal':
        return np.random.randn(m, n) / np.sqrt(m+n)
    return -1


def getActivationValues(layers, neuronInEachLayers):
    '''
    Initialize the activation values (a), weighted sum outputs (z) and error term (δ) for each layer
    All of them have size equal to the "neuorns in current layers + 1"
    '''
    a_values, z_values, δ_values = [], [], []
    for i in range(layers):
        a = np.ones(shape=(neuronInEachLayers[i] + 1, 1))
        z = np.ones(shape=(neuronInEachLayers[i] + 1, 1))
        δ = np.ones(shape=(neuronInEachLayers[i] + 1, 1))
        a_values.append(a)
        z_values.append(z)
        δ_values.append(δ)

    return np.array(a_values), np.array(z_values), np.array(δ_values)
