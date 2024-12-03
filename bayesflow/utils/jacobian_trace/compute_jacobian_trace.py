from collections.abc import Callable
import keras
import numpy as np

from bayesflow.types import Tensor
from ..vjp import vjp


def compute_jacobian_trace(fn: Callable[[Tensor], Tensor], x: Tensor) -> (Tensor, Tensor):
    """Compute the exact trace of the Jacobian matrix of f by projection on each axis.

    :param fn: The function to be differentiated.

    :param x: Tensor of shape (n, ..., d)
        The input tensor to f.

    :return: 2-tuple of tensors:
        1. The output of f(x)
        2. Tensor of shape (n,)
            The exact trace of the Jacobian matrix of f.
    """
    shape = keras.ops.shape(x)
    trace = keras.ops.zeros(shape[:-1])

    fx, vjp_fn = vjp(fn, x)

    for dim in range(shape[-1]):
        projector = np.zeros(shape, dtype="float32")
        projector[..., dim] = 1.0
        projector = keras.ops.convert_to_tensor(projector)

        vjp_value = vjp_fn(projector)[0]

        trace += vjp_value[..., dim]

    return fx, trace
