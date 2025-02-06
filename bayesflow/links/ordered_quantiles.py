import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.utils import keras_kwargs, logging

from collections.abc import Sequence

from .ordered import Ordered


@serializable(package="links.ordered_quantiles")
class OrderedQuantiles(Ordered):
    def __init__(self, q: Sequence[float] = None, axis: int = None, **kwargs):
        super().__init__(axis, None, **keras_kwargs(kwargs))
        self.q = q

        self.config = {
            "q": q,
            "axis": axis,
        }

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    def build(self, input_shape):
        if self.axis is None and 1 < len(input_shape) <= 3:
            self.axis = -2
        elif self.axis is None:
            raise AssertionError(
                f"Cannot resolve which axis should be ordered automatically from input shape {input_shape}."
            )

        num_quantile_levels = input_shape[self.axis]

        if self.q is None:
            # choose the middle of the specified axis as anchor index
            self.anchor_index = num_quantile_levels // 2
            logging.info(
                f"`OrderedQuantiles` was not provided with argument `q`. Using index {self.anchor_index} as anchor."
            )
        else:
            # choose quantile level closest to median as anchor index
            self.anchor_index = keras.ops.argmin(keras.ops.abs(keras.ops.convert_to_tensor(self.q) - 0.5))
            msg = (
                "Length of `q` does not coincide with input shape: "
                f"len(q)={len(self.q)}, position {self.axis} of shape={input_shape}"
            )
            assert num_quantile_levels == len(self.q), msg

        msg = (
            "The link function `OrderedQuantiles` expects at least 3 quantile levels,"
            f" but only {num_quantile_levels} were given."
        )
        assert self.anchor_index not in (0, -1, num_quantile_levels - 1), msg

        super().build(input_shape)
