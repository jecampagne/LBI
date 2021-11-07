import jax
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


def MakeMAF(
    input_dim: int,
    hidden_dim: int = 64,
    context_dim: int = 0,
    n_layers: int = 5,
    context_embedding: nn.Module = None,
    context_embedding_dim: int = 0,
):
    """
    A sequence of affine transformations with a masked affine transform.

    returns init_fun
    """

    if context_embedding is None:
        context_embedding_dim = context_dim
    else:
        context_dim = context_embedding_dim

    made_kwargs = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "context_dim": context_dim,
        "output_dim_multiplier": 2,
    }

    reverse_kwargs = {"input_dim": input_dim}
    random_kwargs = {"input_dim": input_dim, "rng": jax.random.PRNGKey(0)}
    actnorm_kwargs = {}

    reverse = permutations.Reverse
    random = permutations.Random
    actnorm = normalization.ActNorm
    conv = permutations.Conv1x1

    return flow.Flow(
        transformation=utils.SeriesTransform(
            (
                made_module.MADE(**made_kwargs),
                random(**random_kwargs),
                # reverse(**reverse_kwargs),
                # actnorm(**actnorm_kwargs),
            )
            * n_layers,
            context_embedding=context_embedding,
        ),
        prior=priors.Normal(dim=input_dim),
    )


if __name__ == "__main__":
    import jax
    from lbi.models.MLP import MLP
    import optax
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    def loss_fn(params, batch):
        nll = -maf.apply(params, *batch).mean()
        return nll

    def init_fn(rng, input_shape, context_shape=None):
        if context_shape is None:
            context_shape = (0,)
        dummy_input = np.ones((1, *input_shape))
        dummy_context = np.ones((1, *context_shape))
        params = maf.init(rng, dummy_input, context=dummy_context)  # do shape inference
        return params

    def get_train_step(loss_fn, optimizer):
        @jax.jit
        def train_step(params, opt_state, batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return loss, optax.apply_updates(params, updates), opt_state

        return train_step

    # context embedding hyperparams
    context_embedding_kwargs = {
        "output_dim": 4,
        "hidden_dim": 8,
        "num_layers": 1,
        "act": "leaky_relu",
    }

    seed = 1234

    learning_rate = 1e-3
    batch_size = 128
    nsteps = 40

    n_layers = 2
    hidden_dim = 32

    rng = jax.random.PRNGKey(seed)

    # --------------------
    # Create the dataset
    # --------------------
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

    # --------------------
    # Create the model
    # --------------------

    context_embedding = MLP(**context_embedding_kwargs)

    maf = MakeMAF(
        input_dim=input_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        context_embedding=context_embedding,
        context_embedding_dim=context_embedding_kwargs["output_dim"],
    )

    params = init_fn(
        rng=rng,
        input_shape=(input_dim,),
        context_shape=(context_dim,),
    )
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    train_step = get_train_step(loss_fn, optimizer)

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
    samples_0 = maf.apply(
        params, rng, context=np.zeros((1000, context_dim)), method=maf.sample
    )
    plt.scatter(*samples_0.T, color="red", label="0", marker=".", alpha=0.2)
    samples_1 = maf.apply(
        params, rng, context=np.ones((1000, context_dim)), method=maf.sample
    )
    plt.scatter(*samples_1.T, color="blue", label="1", marker=".", alpha=0.2)

    plt.xlim(-1.5, 2.5)
    plt.ylim(-1, 1.5)
    plt.legend()
    plt.show()
