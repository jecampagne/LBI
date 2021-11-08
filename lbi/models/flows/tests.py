import jax
import jax.numpy as np
from maf import construct_MAF


def check_invertibility(seed=42):
    """
    Check if flow is invertible.
    """
    input_dim = 5
    context_dim = 10
    hidden_dim = 64
    n_layers = 5
    rng = jax.random.PRNGKey(seed)

    sample_input = 10 * (jax.random.uniform(rng, (1, input_dim)) - 0.5)
    sample_context = 10 * (jax.random.uniform(rng, (1, context_dim)) - 0.5) 
    
    def init_fn(rng, input_shape, context_shape=None):
        if context_shape is None:
            context_shape = (0,)
        dummy_input = np.ones((1, *input_shape))
        dummy_context = np.ones((1, *context_shape))
        params = maf.init(rng, dummy_input, context=dummy_context)
        return params

    maf = construct_MAF(
        input_dim=input_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        context_embedding=None,
    )

    params = init_fn(
        rng=rng,
        input_shape=(input_dim,),
        context_shape=(context_dim,),
    )

    sample_latent = maf.apply(params, sample_input, sample_context, method=maf.forward)[
        0
    ]
    sample_input_back = maf.apply(
        params, sample_latent, sample_context, method=maf.inverse
    )[0]

    delta = sample_input - sample_input_back
    print("delta", delta)

    if np.allclose(sample_input, sample_input_back):
        print("Flow is invertible")
    else:
        print("Flow is not invertible")
        print("delta std:", np.std(delta))


if __name__ == "__main__":
    check_invertibility()
