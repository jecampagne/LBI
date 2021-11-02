import jax
import jax.numpy as np
import optax
from functools import partial
from .maf import MaskedAffineFlow


def init_fn(input_shape, rng, flow_fns, optimizer):
    dummy_input = np.ones((1, *input_shape))
    params = flow_fns.init(rng, dummy_input)["params"]
    opt_state = optimizer.init(params)
    return params, opt_state


parallel_init_fn = jax.vmap(init_fn, in_axes=(None, 0, None, None), out_axes=(0, 0))


def InitializeFlow(
    model_rng,
    obs_dim,
    theta_dim,
    flow_model=None,
    ensemble_size=5,
    num_layers=5,
    hidden_dim=64,
    context_embedding_kwargs=None,
    **kwargs,
):
    """
    Initialize a flow model.

    Args:
        model_rng: a jax random number generator
        obs_dim: dimensionality of the observations
        theta_dim: dimensionality of the simulation parameters
        n_layers: number of affine layers in the flow

    Returns:
        initial_params: a list of parameters
        log_pdf: a function from parameters to log-probability of the observations
        sample: a function from parameters to samples of the parameters

    """

    def loss(params, inputs, context=None):
        return -log_pdf(params, inputs, context).mean()

    if flow_model is None:
        flow_model = MaskedAffineFlow
    if type(model_rng) is int:
        model_rng = jax.random.PRNGKey(model_rng)

    ensemble_seeds = jax.random.split(model_rng, ensemble_size)

    if context_embedding_kwargs is None:
        context_embedding_kwargs = {
            "use_context_embedding": False,
            "embedding_dim": None,
        }

    # init_fn = flow_model(num_layers, context_embedding_kwargs=context_embedding_kwargs)
    # parallel_init_fn = jax.vmap(init_fn, in_axes=(0,), out_axes=(0,None, None))

    initial_params_vector, opt_state_vector = parallel_init_fn(
        ensemble_seeds,
        input_dim=obs_dim,
        context_dim=theta_dim,
        hidden_dim=hidden_dim,
    )

    return loss, (log_pdf, sample), initial_params_vector, opt_state_vector
