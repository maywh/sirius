from numpy import genfromtxt

from sirius.model import Model
from sirius.layers import HiddenLayer

data = genfromtxt('tests/data/iris.csv', delimiter=',', skip_header=1)
model = Model(input_data = data)
layer1 = HiddenLayer(dimensions=len(data), units=2, activation='sigmoid')

def test_model_init():
    assert model._depth == 0
    assert not model._network


def test_model_add():
    model.add(layer1)
    assert model._depth == 1
    assert len(model._network) == 1
    