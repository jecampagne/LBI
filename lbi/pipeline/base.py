import jax
import jax.numpy as np
import numpy as onp
import optax
from lbi.dataset import getDataLoaderBuilder
from lbi.sequential.sequential import sequential
from lbi.models import parallel_init_fn
from lbi.models.steps import get_train_step, get_valid_step
from lbi.models.flows import construct_MAF
from lbi.models.MLP import MLP
from lbi.models.classifier import construct_Classifier
from lbi.models.classifier.classifier import get_loss_fn
from lbi.trainer import getTrainer


def pipeline(
    rng,
    X_true,
    get_simulator,
    # Prior
    log_prior,
    sample_prior,
    # Simulator
    simulator_kwargs={},
    # Model hyperparameters
    model_type="classifier",  # "classifier" or "flow"
    ensemble_size=15,
    num_layers=2,
    hidden_dim=32,
    # classifier parameters
    use_residual=False,
    # flow specific parameters
    transform_type="MaskedPiecewiseRationalQuadraticAutoregressiveTransform",
    permutation="Reverse",
    tail_bound=10.0, 
    num_bins=10,
    # Optimizer hyperparmeters
    max_norm=1e-3,
    learning_rate=3e-4,
    weight_decay=1e-1,
    # Train hyperparameters
    nsteps=250000,
    patience=15,
    eval_interval=100,
    # Dataloader hyperparameters
    batch_size=32,
    train_split=0.8,
    num_workers=0,
    sigma=None,
    add_noise=False,
    scale_X=None,
    scale_Theta=None,
    # Sequential hyperparameters
    num_rounds=3,
    num_initial_samples=1000,
    num_warmup_per_round=100,
    num_samples_per_round=100,
    num_chains=10,
    logger=None,
):
    """
    Takes in logger, LBI parameters and get_simulator() function
    then runs the LBI pipeline. Returns log_likelihood, log_prior,
    trained ensemble parameters, and samples generated.
    """

    # --------------------------
    # set up simulation and observables
    simulate, obs_dim, theta_dim = get_simulator(**simulator_kwargs)

    data_loader_builder = getDataLoaderBuilder(
        sequential_mode=model_type,
        batch_size=batch_size,
        train_split=train_split,
        num_workers=num_workers,
        sigma=sigma,
        add_noise=add_noise,
    )
    # TODO: Package model, optimizer, trainer initialization into a function

    # --------------------------
    # Create optimizer
    optimizer = optax.chain(
        # Set the parameters of Adam optimizer
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
        ),
        optax.adaptive_grad_clip(max_norm),
    )

    # --------------------------
    # Create model
    if model_type == "classifier":

        classifier_kwargs = {
            # "output_dim": 1,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "use_residual": False,
            "act": "gelu",
        }
        model, log_prob, loss_fn = construct_Classifier(
            scale_X=scale_X, 
            scale_Theta=scale_Theta, 
            **classifier_kwargs
        )
    else:
        maf_kwargs = {
            "rng": rng,
            "transform_type": transform_type,
            "input_dim": obs_dim,
            "hidden_dim": hidden_dim,
            "context_dim": theta_dim,
            "n_layers": num_layers,
            "n_bins": num_bins,
            "tail_bound": tail_bound,
            "permutation": permutation,
            "normalization": None,
            "made_activation": "gelu",
        }
        context_embedding_kwargs = {
            "output_dim": theta_dim * 2,
            "hidden_dim": theta_dim * 2,
            "num_layers": 2,
            "act": "leaky_relu",
        }

        context_embedding = MLP(**context_embedding_kwargs)
        model, log_prob, loss_fn = construct_MAF(
            context_embedding=context_embedding,
            scale_X=scale_X,
            scale_Theta=scale_Theta,
            **maf_kwargs
        )

    params, opt_state = parallel_init_fn(
        jax.random.split(rng, ensemble_size),
        model,
        optimizer,
        (obs_dim,),
        (theta_dim,),
    )

    parallel_log_prob = jax.vmap(log_prob, in_axes=(0, None, None))
    # --------------------------
    # Create trainer

    train_step = get_train_step(loss_fn, optimizer)
    valid_step = get_valid_step({"valid_loss": loss_fn})

    trainer = getTrainer(
        train_step,
        valid_step=valid_step,
        nsteps=nsteps,
        eval_interval=eval_interval,
        patience=patience,
        logger=logger,
        train_kwargs=None,
        valid_kwargs=None,
    )

    # Train model sequentially
    params, Theta_post = sequential(
        rng,
        X_true,
        params,
        parallel_log_prob,
        log_prior,
        sample_prior,
        simulate,
        opt_state,
        trainer,
        data_loader_builder,
        num_rounds=num_rounds,
        num_initial_samples=num_initial_samples,
        num_samples_per_round=num_samples_per_round,
        num_samples_per_theta=1,
        num_chains=num_chains,
        logger=logger,
    )

    return model, log_prob, params, Theta_post
