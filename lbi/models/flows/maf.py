# from re import A
from flax.linen.module import compact
import jax.numpy as np
import flax.linen as nn

import flow
import priors
import made as made_module
import permutations
import normalization
import utils 

from typing import Any

Distribution = Any


def MaskedAffineFlow(n_layers=5, context_embedding_kwargs=None):
    """
    A sequence of affine transformations with a masked affine transform.

    returns init_fun
    """
    made = made_module.MADE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        output_dim_multiplier=2,
    )
    reverse = permutations.Reverse(input_dim=input_dim)
    actnorm = normalization.ActNorm()

    return flow.Flow(
        transformation=utils.SeriesTransform(
            (
                made,
                reverse,
                # actnorm,
            ),
            *n_layers
        ),
        prior=priors.Normal(),
        context_embedding_kwargs=context_embedding_kwargs,
    )


class MaskedAffineFlow(nn.Module):
    input_dim: int
    n_layers: int
    prior: Distribution

    def setup(self):
        return super().setup()

    @compact
    def apply(self, x, context=None):
        pass


if __name__ == "__main__":
    import jax
    import optax
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from jax.experimental import stax
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    def loss(params, inputs, context=None):
        return -log_pdf(params, inputs, context).mean()

    @jax.jit
    def train_step(params, opt_state, batch):
        nll, grads = jax.value_and_grad(loss)(params, *batch)
        updates, opt_state = opt_update(grads, opt_state, params)
        return nll, optax.apply_updates(params, updates), opt_state

    hidden_dim = 32

    n_layers = 4

    # context embedding hyperparams
    context_embedding_kwargs = {
        "use_context_embedding": True,
        "embedding_dim": 16,
        "hidden_dim": 128,
        "num_layers": 2,
        "act": "celu",
    }

    batch_size = 128
    seed = 1234
    nsteps = 40

    X, y = make_moons(n_samples=10000, noise=0.05, random_state=seed)
    y = y[:, None]
    input_dim = X.shape[1]
    context_dim = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, stratify=y, random_state=seed
    )

    X_train_s = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    train_dataloader = DataLoader(
        TensorDataset(X_train_s, y_train), batch_size=batch_size, shuffle=True
    )

    rng = jax.random.PRNGKey(seed)
    params, log_pdf, sample = MaskedAffineFlow(
        n_layers=n_layers,
        context_embedding_kwargs=context_embedding_kwargs,
    )(
        rng,
        input_dim=input_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
    )

    learning_rate = 1e-4
    opt_init, opt_update = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.adamw(learning_rate=learning_rate),
    )

    opt_state = opt_init(params)

    iterator = tqdm(range(nsteps))
    try:
        for _ in iterator:
            for batch in train_dataloader:
                batch = [np.array(a) for a in batch]
                nll, params, opt_state = train_step(params, opt_state, batch)
            iterator.set_description("nll = {:.3f}".format(nll))
    except KeyboardInterrupt:
        pass

    plt.scatter(*X_train.T, color="grey", alpha=0.01, marker=".")

    samples_0 = sample(rng, params, context=np.zeros((1000, context_dim)))
    plt.scatter(*samples_0.T, color="red", label="0", marker=".", alpha=0.2)
    samples_1 = sample(rng, params, context=np.ones((1000, context_dim)))
    plt.scatter(*samples_1.T, color="blue", label="1", marker=".", alpha=0.2)

    plt.xlim(-1.5, 2.5)
    plt.ylim(-1, 1.5)
    plt.legend()
    plt.show()
