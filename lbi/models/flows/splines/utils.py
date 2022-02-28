import jax.numpy as np


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations = bin_locations.at[..., -1].add(eps)
    return np.sum(inputs[..., None] >= bin_locations, axis=-1) - 1


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    reduce_dims = list(range(num_batch_dims, len(x.shape)))
    return np.sum(x, axis=reduce_dims)
