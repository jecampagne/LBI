import jax
import jax.numpy as np

import flax.linen as nn


class ResidualBlock(nn.Module):
    hidden_dim: int
    act: str = "celu"

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.hidden_dim)(x)
        y = getattr(nn, self.act)(y)
        y = nn.Dense(x.shape[-1])(y)
        return x + y


class MLP(nn.Module):
    # TODO: replace with Sequential
    output_dim: int
    num_layers: 5
    hidden_dim: 128
    use_residual: bool = False
    act: str = "celu"

    @nn.compact
    def __call__(self, *args):
        x = np.hstack(args)
        for layer in range(self.num_layers):
            if self.use_residual:
                x = ResidualBlock(self.hidden_dim, self.act)(x)
            else:
                x = nn.Dense(self.hidden_dim)(x)
                x = getattr(nn, self.act)(x)
        return nn.Dense(self.output_dim)(x)


class GammaResidualMLP(nn.Module):
    """
    This is an MLP that appends the gamma parameter to every
    layer's output.

    NB: It assumes the first argument's first gamma_dim
    elements are the gamma parameters.
    """

    gamma_dim: int
    output_dim: int
    num_layers: 5
    hidden_dim: 128
    use_residual: bool = False
    act: str = "celu"

    @nn.compact
    def __call__(self, *args):
        x = np.hstack(args)
        gamma = x[..., : self.gamma_dim]

        for layer in range(self.num_layers):
            if self.use_residual:
                x = ResidualBlock(self.hidden_dim, self.act)(x)
            else:
                x = nn.Dense(self.hidden_dim)(x)
                x = getattr(nn, self.act)(x)
            x = np.hstack([gamma, x])
        # we subtract gamma_dim from the output_dim to preserve
        # the output shape
        out = nn.Dense(self.output_dim - self.gamma_dim)(x)  
        return np.hstack([gamma, out])
