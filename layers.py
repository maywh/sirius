import numpy as np

class Layer(object):
    # Parent layer
    def __init__(self):
        pass


class HiddenLayer(Layer):
    # NN hidden layer
    def __init__(self, activation, n_neurons, weight_initialisation='uniform'):
        self._weight_inilization = weight_initialisation
        self._activation = activation

    def _initialise_weights(self, dim, weight_initialisation, n_neurons):
        if weight_initialisation == 'uniform':
            weights = np.random.uniform(0, 0.05, (dim, n_neurons))
        return weights

    def _inilitalise_bias(self, n_neurons):
        bias = np.zeros(n_neurons)
        return bias

    def _activation_function(self, activation):
        if activation == "sigmoid":
            # activated = 1 / (1 + np.exp(-product)

            return None


class OutputLayer(Layer):
    # NN output layer
    pass


class Dropout(Layer):
    pass
