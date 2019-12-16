import numpy as np

from sirius.layers import HiddenLayer


class Model:

    
    def __init__(self, input_data):
        self._input_data = input_data
        self._m = input_data.shape[0]
        self.depth = 0
        self._network = []

      
    def add(self, layer, **kwargs):

        if self.depth == 0:
            assert layer.shape == self._m
        else:
            layer.shape = _network[depth-1].shape

        self._network.append(layer)
        self.depth += 1


    def forward_propagation(self):
        A = self._forward_propagate(self._input_data, self._network[0])
        for i in range(1, -self.depth):
            A = self._forward_propagate(A, self._network[i])
        return A


    def back_propagation(self):
        pass


    def _forward_propagate(self, Z, layer):
        Z = np.dot(Z, layer.weights) + layer.bias
        A = layer.activate_and_update_attributes(layer, Z)
        return A

    
    def _backward_propagate(self, layer):
        pass


    def cost(self, Y, Y_hat):
        return -1/self._m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1-Y, np.log(Y_hat).T))
        
