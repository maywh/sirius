import numpy as np

from sirius.layers import HiddenLayer


class Model:

    _depth = 0
    _network = []
    _input_data = []
    
    def __init__(self, input_data):
        self._input_data = input_data
      

    def add(self, activation, n_neurons, **kwargs):

        if self._depth == 0:
            n_input = self._input_data.shape[0]
        # else:
        #     n_input = _network[_depth-1].

        layer = HiddenLayer(activation = activation, n_input = n_input, n_neurons=n_neurons)
        self._network.append(layer)
        self._depth += 1

    def forward_propagation(self):
        z = self._propagate(self._input_data, self._network[0])
        for i in range(1, self.depth):
            z = self._propagate(z, self._network[i])

    def back_propagation(self):
        pass

    def _propagate(self, input_layer, layer):
        z = np.dot(input_layer, layer.weights) + layer.bias
        z = self._activate(layer, z)
        return z

    def _activate(self, layer, z):
        return layer.activate(z)

    def cost(self):
        pass
        
