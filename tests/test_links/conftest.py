import numpy as np
import keras
import pytest


@pytest.fixture()
def batch_size():
    return 16


@pytest.fixture()
def num_variables():
    return 10


@pytest.fixture()
def generic_preactivation(batch_size):
    return keras.ops.ones((batch_size, 4, 4))


@pytest.fixture()
def ordered():
    from bayesflow.links import Ordered

    return Ordered(axis=1, anchor_index=2)


@pytest.fixture()
def ordered_quantiles():
    from bayesflow.links import OrderedQuantiles

    return OrderedQuantiles()


@pytest.fixture()
def positive_semi_definite():
    from bayesflow.links import PositiveSemiDefinite

    return PositiveSemiDefinite()


@pytest.fixture()
def linear():
    return keras.layers.Activation("linear")


@pytest.fixture(params=["ordered", "ordered_quantiles", "positive_semi_definite", "linear"], scope="function")
def link(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def num_quantiles():
    return 19


@pytest.fixture()
def quantiles_np(num_quantiles):
    return np.linspace(0, 1, num_quantiles + 2)[1:-1]


@pytest.fixture()
def quantiles_py(quantiles_np):
    return list(quantiles_np)


@pytest.fixture()
def quantiles_keras(quantiles_np):
    return keras.ops.convert_to_tensor(quantiles_np)


@pytest.fixture()
def none():
    return None


@pytest.fixture(params=["quantiles_np", "quantiles_py", "quantiles_keras", "none"], scope="function")
def quantiles(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def unordered(batch_size, num_quantiles, num_variables):
    return keras.random.normal((batch_size, num_quantiles, num_variables))


@pytest.fixture()
def random_matrix_batch(batch_size, num_variables):
    return keras.random.normal((batch_size, num_variables, num_variables))
