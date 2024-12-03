from collections.abc import Callable
import keras

from bayesflow.types import Tensor


def jvp(fn: Callable, primals: Tensor | tuple[Tensor, ...], tangents: Tensor | tuple[Tensor, ...]) -> (any, Tensor):
    """
    Backend-agnostic version of the Jacobian-vector product (jvp).
    Compute the Jacobian-vector product of the given function at the point given by the input (primals).

    :param fn: The function to differentiate.
        Signature and return value must be compatible with the vjp method of the backend in use.

    :param primals: Input tensors to `fn`.

    :param tangents: Tangent vectors to differentiate `fn` with respect to.

    :return: The output of `fn(*primals)` and the Jacobian-vector product of `fn` evaluated at `primals` with respect to
        `tangents`.
    """

    match keras.backend.backend():
        case "jax":
            import jax

            fx, _jvp = jax.jvp(fn, primals, tangents)
        case "torch":
            import torch

            fx, _jvp = torch.func.jvp(fn, primals, tangents)
        case "tensorflow":
            import tensorflow as tf

            with tf.autodiff.ForwardAccumulator(primals, tangents) as acc:
                fx = fn(*primals)

            _jvp = acc.jvp(fx)
        case _:
            raise NotImplementedError(f"JVP not implemented for backend {keras.backend.backend()}")

    return fx, _jvp
