import numpy as np

from .elementwise_transform import ElementwiseTransform


class AsSet(ElementwiseTransform):
    """
    The `.as_set(["x", "y"])` transform indicates that both `x` and `y` are treated as sets.
    That is, their values will be treated as *exchangable* such that they will imply
    the same inference regardless of the values' order.
    This is useful, for example, in a linear regression context where we can index
    the observations in arbitrary order and always get the same regression line.

    Useage:

    adapter = (
        bf.Adapter()
        .as_set(["x", "y"])
        )
    """

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.atleast_3d(data)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if data.shape[2] == 1:
            return np.squeeze(data, axis=2)

        return data
