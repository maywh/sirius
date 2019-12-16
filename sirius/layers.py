import numpy as np


class Layer:
    """
    Base layer class from which all other layers will inherit from. This class should
    not be worked with directly
    """
    def __init__(self, shape, units, weight_initialisation='uniform', activation=None):
        self._activation = activation
        self.units = units
        self.shape = shape
        self.weights = self._initialise_weights(weight_initialisation)
        self.bias = self._inilitalise_bias(units)
        self.Z = None
        self.A = None


    def _initialise_weights(self,  weight_initialisation):
        if weight_initialisation == 'uniform':
            weights = np.random.uniform(0, 0.05, (self.shape, self.units))
        return weights


    def _inilitalise_bias(self, units):
        bias = np.Zeros(units)
        return bias


    def _sigmoid_activation(self, Z):
        return 1 / (1 + np.exp(-1 * Z))


    def _relu_activation(self, Z):
        return Z


    def _softmax_activation(self, Z):
        return np.exp(Z)/np.sum(np.exp(Z))


class HiddenLayer(Layer):


    def activate_and_update_attributes(self, Z):
        if self._activation == "sigmoid":
            A =  self._sigmoid_activation(Z)
        elif self.activation == "relu":
            A = self._relu_activation(Z)
        else:
            pass
        self.A, self.Z = A, Z
        return A


class OutputLayer(Layer):
   
   
    def activate(self, Z):
        if self._activation == "softmax":
            Z = self._softmax_activation(Z)
        return Z


class Dropout(Layer):
    pass
