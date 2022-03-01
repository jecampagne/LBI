import jax
import jax.numpy as np
import flax.linen as nn

from lbi.models.flows import flow, priors, utils, permutations, normalizations
import lbi.models.flows.made as made_module
from lbi.models.flows import transforms


def get_loss_fn(log_pdf):
    def loss_fn(params, *args):
        """
        Negative-log-likelihood loss function.

        *args: inputs, context (optional)
        """
        return -log_pdf(params, *args).mean()

    return loss_fn


def construct_MAF(
    rng: jax.random.PRNGKey,
    input_dim: int,
    hidden_dim: int = 64,
    context_dim: int = 0,
    n_layers: int = 5,
    n_bins: int = 8,
    context_embedding: nn.Module = None,
    permutation: str = "Conv1x1",
    normalization: str = None,
    transform_type: str = "MaskedLinearAutoregressiveTransform",
    made_activation: str = "gelu",
    tail_bound: float = 6.0,
    scale_X=None,
    scale_Theta=None,
):
    """
    A sequence of affine transformations with a masked affine transform.
    """

    if scale_X is None:
        scale_X = lambda x: x
    if scale_Theta is None:
        scale_Theta = lambda x: x
    if context_embedding is not None:
        context_dim = context_embedding.output_dim

    # made_kwargs = {
    #     "input_dim": input_dim,
    #     "hidden_dim": hidden_dim,
    #     "context_dim": context_dim,
    #     "output_dim_multiplier": 2,
    #     "act": made_activation,
    # }

    piecewise_kwargs = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "context_dim": context_dim,
        "num_bins": n_bins,
        "act": made_activation,
        "tail_bound": tail_bound,
    }

    permutation = getattr(permutations, permutation)
    permutation_kwargs = {"input_dim": input_dim, "rng": None}

    if normalization is not None:
        normalization = getattr(normalizations, normalization)
    normalization_kwargs = {}

    transformations = []
    for rng in jax.random.split(rng, n_layers):
        permutation_kwargs["rng"] = rng
        transformations.append(getattr(transforms, transform_type)(**piecewise_kwargs))

        # transformations.append(made_module.MADE(**made_kwargs))
        # transformations.append(MaskedLinearAutoregressiveTransform(**piecewise_kwargs))
        # transformations.append(
        #     MaskedPiecewiseLinearAutoregressiveTransform(**piecewise_kwargs)
        # )
        # transformations.append(
        #     MaskedPiecewiseRationalQuadraticAutoregressiveTransform(**piecewise_kwargs)
        # )
        transformations.append(permutation(**permutation_kwargs))
        if normalization is not None:
            transformations.append(normalization(**normalization_kwargs))

    maf = flow.Flow(
        transformation=utils.SeriesTransform(
            transformations=transformations,
            context_embedding=context_embedding,
        ),
        prior=priors.Normal(dim=input_dim),
    )

    def log_prob_fn(params, X, Theta=None):
        # print("X shape: ", X.shape)
        # print("Theta shape: ", Theta.shape)

        scaled_X = scale_X(X) if scale_X is not None else X
        if Theta is not None:
            scaled_Theta = scale_Theta(Theta) if scale_Theta is not None else Theta
        else:
            scaled_Theta = None

        # the models' __call__ are their log_prob fns
        return maf.apply(params, scaled_X, scaled_Theta)

    return maf, log_prob_fn, get_loss_fn(log_prob_fn)
