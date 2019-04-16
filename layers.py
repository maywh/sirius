import numpy as np


class Layer:
    # Parent layer
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

    def _sigmoid_activation(self, product):
        return 1 / (1 + np.exp(-1 * product))

    def _relu_activation(self, product):
        return product

    def _softmax_activation(self, product):
        return np.exp(product)/np.sum(np.exp(product))


class HiddenLayer:
    # NN hidden layer
    def __init__(self, activation, n_input, n_neurons, weight_initialisation='uniform'):
        super().__init__(n_input, n_neurons, weight_initialisation)
        self._activation = activation

    def _activate(self, product):
        if self._activation == "sigmoid":
            z =  self._sigmoid_activation(product)
        elif self.activation == "relu":
            z =  self._relu_activation(product)
        else:
            pass
        return z


class OutputLayer:
    # NN output layer
    def __init__(self, activation, n_output, n_neurons, weight_initialisation='uniform'):
        super().__init__(n_output, n_neurons, weight_initialisation)
        self.n_output = n_output
        self._activation = activation

    def _activate(self, product):
        if self._activation == "softmax":
            z = self._sigmoid_activation(product)
        return z


class Dropout(Layer):
    pass
