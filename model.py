import numpy as np

class Model:
    
    def __init__(self, input_data):
        self._input_data = input_data
        self._network = []
        self.depth = 0

    def add(self, layer):
        self._network.append(layer)
        self.depth += 1

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

    def _activate(self, layer, product):
        return layer.activate(product)
        
