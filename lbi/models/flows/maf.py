import jax
import jax.numpy as np
import flax.linen as nn

import flow
import priors
import made as made_module
import permutations
import normalizations
import utils


def construct_MAF(
    rng: jax.random.PRNGKey,
    input_dim: int,
    hidden_dim: int = 64,
    context_dim: int = 0,
    n_layers: int = 5,
    context_embedding: nn.Module = None,
    permutation: str = "Reverse",
    normalization: str = None,
):
    """
    A sequence of affine transformations with a masked affine transform.

    returns init_fun
    """

    if context_embedding is not None:
        context_dim = context_embedding.output_dim

    made_kwargs = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "context_dim": context_dim,
        "output_dim_multiplier": 2,
    }

    permutation = getattr(permutations, permutation)
    permutation_kwargs = {"input_dim": input_dim, "rng": None}

    if normalization is not None:
        normalization = getattr(normalizations, normalization)
    normalization_kwargs = {}

    transformations = []
    for rng in jax.random.split(rng, n_layers):
        permutation_kwargs["rng"] = rng

        transformations.append(made_module.MADE(**made_kwargs))
        transformations.append(permutation(**permutation_kwargs))
        if normalization is not None:
            transformations.append(normalization(**normalization_kwargs))

    return flow.Flow(
        transformation=utils.SeriesTransform(
            transformations=transformations,
            context_embedding=context_embedding,
        ),
        prior=priors.Normal(dim=input_dim),
    )


if __name__ == "__main__":
    import jax
    from lbi.models.MLP import MLP
    from lbi.models.steps import get_train_step
    import optax
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    def loss_fn(params, *args):
        nll = -maf.apply(params, *args).mean()
        return nll

    def init_fn(rng, input_shape, context_shape=None):
        if context_shape is None:
            context_shape = (0,)
        dummy_input = np.ones((1, *input_shape))
        dummy_context = np.ones((1, *context_shape))
        params = maf.init(rng, dummy_input, context=dummy_context)  # do shape inference
        return params


    seed = 1234
    rng = jax.random.PRNGKey(seed)
    
    learning_rate = 1e-3
    batch_size = 128
    nsteps = 40

    n_layers = 1
    hidden_dim = 128



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

    maf_kwargs = {
        "rng": rng,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "context_dim": context_dim,
        "n_layers": n_layers,
        "permutation": "Reverse",
        "normalization": None,
    }
    # context embedding hyperparams
    context_embedding_kwargs = {
        "output_dim": 4,
        "hidden_dim": 8,
        "num_layers": 1,
        "act": "leaky_relu",
    }


    context_embedding = MLP(**context_embedding_kwargs)
    maf = construct_MAF(context_embedding=context_embedding, **maf_kwargs)

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
