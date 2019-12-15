import numpy as np


class Layer:
    """
    Base layer class from which all other layers will inherit from. This class should
    not be worked with directly
    """
    def __init__(self, dimensions, units, weight_initialisation='uniform', activation=None):
        self.weights = self._initialise_weights(dimensions, units, weight_initialisation)
        self.bias = self._inilitalise_bias(units)
        self._activation=activation


    def _initialise_weights(self, dimensions, units, weight_initialisation):
        if weight_initialisation == 'uniform':
            weights = np.random.uniform(0, 0.05, (dimensions, units))
        return weights


    def _inilitalise_bias(self, units):
        bias = np.zeros(units)
        return bias


    def _sigmoid_activation(self, z):
        return 1 / (1 + np.exp(-1 * z))


    def _relu_activation(self, z):
        return z


    def _softmax_activation(self, z):
        return np.exp(z)/np.sum(np.exp(z))


class HiddenLayer(Layer):


    def activate(self, z):
        if self._activation == "sigmoid":
            z =  self._sigmoid_activation(z)
        elif self.activation == "relu":
            z =  self._relu_activation(z)
        else:
            pass
        return z


class OutputLayer(Layer):
   
   
    def activate(self, z):
        if self._activation == "softmax":
            z = self._sigmoid_activation(z)
        return z


class Dropout(Layer):
    pass
