import jax
import jax.numpy as np
import flax.linen as nn


class Reverse(nn.Module):
    input_dim: int

    def __call__(self, inputs, *args, **kwargs):
        perm = np.arange(inputs.shape[-1])[::-1]
        return inputs[:, perm], np.zeros(inputs.shape[:1])

    def forward(self, inputs, *args, **kwargs):
        return self(inputs, *args, **kwargs)

    def inverse(self, inputs, *args, **kwargs):
        perm = np.arange(inputs.shape[-1])[::-1]
        return inputs[:, perm], np.zeros(inputs.shape[:1])


class Random(nn.Module):
    """
    Probably best to use different rng's for each permutation
    """
    rng: jax.random.Generator
    input_dim: int

    def setup(self):
        self.perm = jax.random.permutation(self.rng, np.arange(self.input_dim))

    def __call__(self, inputs, *args, **kwargs):
        return inputs[:, self.perm], np.zeros(inputs.shape[:1])

    def forward(self, inputs, *args, **kwargs):
        self(inputs, *args, **kwargs)

    def inverse(self, inputs, *args, **kwargs):
        inverse_perm = np.argsort(self.perm)
        return inputs[:, inverse_perm], np.zeros(inputs.shape[:1])
