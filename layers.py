import numpy as np

class Layer(object):
    # Parent layer
    def __init__(self):
        pass


class HiddenLayer(Layer):
    # NN hidden layer
    def __init__(self, activation, weight_initialisation='uniform'):
        self._weight_inilization = weight_initialisation
        self._activation = activation

    def initialise_weights(dim, weight_initialisation):
        if weight_initialisation == 'uniform':
            weights = np.random.uniform(0, 0.05, dim)

        return weights

    def activation_function(activation):
        if activation == "sigmoid":
            # activated = 1 / (1 + np.exp(-product)

            return None


class OutputLayer(Layer):
    # NN output layer
    pass


class Dropout(Layer):
    pass
