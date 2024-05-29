
import keras
import pytest


@pytest.fixture(params=[2, 3])
def batch_size(request):
    return request.param


@pytest.fixture()
def coupling_flow():
    from bayesflow.experimental.networks import CouplingFlow
    return CouplingFlow.new()


@pytest.fixture(params=["coupling_flow"])
def inference_network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["inference_network", "summary_network"])
def network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[2, 3])
def num_features(request):
    return request.param


@pytest.fixture()
def random_samples(batch_size, num_features):
    return keras.random.normal()


@pytest.fixture()
def random_set(batch_size, set_size, num_features):
    return keras.random.normal((batch_size, set_size, num_features))


@pytest.fixture()
def resnet():
    from bayesflow.experimental.networks import ResNet
    return ResNet.new()


@pytest.fixture(params=[2, 3])
def set_size(request):
    return request.param


@pytest.fixture(params=[])
def summary_network(request):
    return request.getfixturevalue(request.param)
