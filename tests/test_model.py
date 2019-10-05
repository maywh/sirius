from numpy import genfromtxt

from sirius import model

data = genfromtxt('tests/data/iris.csv', delimiter=',', skip_header=1)
model = model.Model(input_data = data)

def test_model_init():
    assert model._depth == 0
    assert not model._network


def test_model_add():
    model.add(activation="sigmoid", n_neurons=2)
    assert model._depth == 1
    assert len(model._network) == 1
    