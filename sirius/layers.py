import numpy as np


class Layer:
    # Parent layer
    weights = []
    bias = []

    def __init__(self, n_target, n_neurons, weight_initialisation):
        self.weights = self._initialise_weights(n_target, n_neurons, weight_initialisation)
        self.bias = self._inilitalise_bias(n_neurons)

    def _initialise_weights(self, n_target, n_neurons, weight_initialisation):
        if weight_initialisation == 'uniform':
            weights = np.random.uniform(0, 0.05, (n_target, n_neurons))
        return weights

    def _inilitalise_bias(self, n_neurons):
        bias = np.zeros(n_neurons)
        return bias

    def _sigmoid_activation(self, z):
        return 1 / (1 + np.exp(-1 * z))

    def _relu_activation(self, z):
        return z

    def _softmax_activation(self, z):
        return np.exp(z)/np.sum(np.exp(z))


class HiddenLayer(Layer):
    # NN hidden layer
    def __init__(self, activation, n_input, n_neurons, weight_initialisation='uniform'):
        super().__init__(n_input, n_neurons, weight_initialisation)
        self._activation = activation

    def activate(self, z):
        if self._activation == "sigmoid":
            z =  self._sigmoid_activation(z)
        elif self.activation == "relu":
            z =  self._relu_activation(z)
        else:
            pass
        return z


class OutputLayer(Layer):
    # NN output layer
    def __init__(self, activation, n_output, n_neurons, weight_initialisation='uniform'):
        super().__init__(n_output, n_neurons, weight_initialisation)
        self.n_output = n_output
        self._activation = activation

    def activate(self, z):
        if self._activation == "softmax":
            z = self._sigmoid_activation(z)
        return z


class Dropout(Layer):
    pass
