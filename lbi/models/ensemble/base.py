import jax
import jax.numpy as np
from jax.experimental import stax


def CreateEncoder(output_dim=32, hidden_dim=128, num_layers=2, act=None):
    if act is None:
        act = stax.Selu

    layers = [lyr for _ in range(num_layers) for lyr in (stax.Dense(hidden_dim), act)]
    layers += [stax.Dense(output_dim)]
    
    init_random_params, embed = stax.serial(*layers)

    return init_random_params, embed
