import numpy as np


class Layer(object):
    # Parent layer
    def __init__(self, n_target, n_neurons, weight_initialisation):
        self.weights = self.initialise_weights(n_target, n_neurons, weight_initialisation)

    def _initialise_weights(self, n_target, n_neurons, weight_initialisation):
        if weight_initialisation == 'uniform':
            weights = np.random.uniform(0, 0.05, (n_target, n_neurons))
        return weights


class HiddenLayer(Layer):
    # NN hidden layer
    def __init__(self, activation, n_input, n_neurons, weight_initialisation='uniform'):
        self._weights = super().__init__(n_input, n_neurons, weight_initialisation)
        self._activation = activation

    def _inilitalise_bias(self, n_neurons):
        bias = np.zeros(n_neurons)
        return bias

    def _activation_function(self, activation):
        if activation == "sigmoid":
            # activated = 1 / (1 + np.exp(-product)

            return None


class OutputLayer(Layer):
    # NN output layer
    def __init__(self, n_output, n_neurons, weight_initialisation='uniform'):
        self._weights = super().__init__(n_output, n_neurons, weight_initialisation)
        self.n_output = n_output


class Dropout(Layer):
    pass
