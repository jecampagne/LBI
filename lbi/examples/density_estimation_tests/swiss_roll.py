import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from lbi.models import parallel_init_fn
from lbi.models.flows import construct_MAF
from lbi.models.steps import get_train_step, get_valid_step
import jax.numpy as np
import numpy as onp 
import jax 
import optax
from tqdm.auto import tqdm



# --------------------------

ensemble_size = 10
seed = 1234
rng, model_rng, hmc_rng = jax.random.split(jax.random.PRNGKey(seed), num=3)

logger = None
# --------------------------

def simulate(num_samples, noise=1.75):
    global seed 
    X, _ = make_swiss_roll(n_samples=num_samples, noise=noise, random_state=seed)
    seed += 1
    X = np.delete(X, 1, axis=1)
    return X/5


# --------------------------

seed = 1234
rng, model_rng, hmc_rng = jax.random.split(jax.random.PRNGKey(seed), num=3)


# --------------------------
# Create optimizer
optimizer = optax.chain(
    # Set the parameters of Adam optimizer
    optax.adamw(
        learning_rate=1e-3,
        weight_decay=1e-3,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
    ),
    optax.adaptive_grad_clip(1e-4),
)

maf_kwargs = {
    "rng": model_rng,
    "input_dim": 2,
    "hidden_dim": 64,
    "context_dim": 0,
    "n_layers": 2,
    "n_bins": 5,
    "permutation": "Reverse",
    "normalization": None,
    "made_activation": "gelu",
}

model, log_prob, loss_fn = construct_MAF(
    **maf_kwargs
)

train_step = get_train_step(loss_fn, optimizer)
valid_step = get_valid_step({"valid_loss": loss_fn})

params, opt_state = parallel_init_fn(
    jax.random.split(rng, ensemble_size),
    model,
    optimizer,
    (2,),
    None,
)
    
iterator = tqdm(range(20000))
try:
    for _step_num in iterator:
        batch = [simulate(128)]
        nll_vector, params, opt_state = train_step(
            params,
            opt_state,
            batch,
        )
        iterator.set_description("mean train loss: %s" % np.mean(nll_vector))
except KeyboardInterrupt:
    print("\nTraining keyboard interrupted")
        
def sample(rng, params, context=None, num_samples=1):
    return model.apply(params, rng, num_samples, context, method=model.sample)

# Create parallelized functions
parallel_log_prob = jax.vmap(log_prob, in_axes=(0, None, None))
parallel_sample = jax.vmap(sample, in_axes=(0, 0, None, None))

def ensemble_sample(rng, params, context=None, num_samples=1):
    """
    When sampling from an ensemble, we need to sample from each model in the ensemble
    and then choose one sample per context from the ensemble. This choice is a sample
    from the weighted distribution over the models (the models are equally weighted so
    this is just a uniform choice).
    """
    all_samples = parallel_sample(rng, params, context, num_samples)
    
    choice_key = jax.random.split(rng[0], 2)[0]
    n, m, d = all_samples.shape
    sample_indices = jax.random.randint(choice_key, shape=(m,), minval=0, maxval=n)
    
    return all_samples[sample_indices, np.arange(m)]


flow_samples = ensemble_sample(
    jax.random.split(model_rng, ensemble_size),
    params.slow if hasattr(params, "slow") else params,
    None, 
    5000
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].scatter(*simulate(100, noise=0).T, marker='.', alpha=0.2, color='k')
axes[0].scatter(*flow_samples.T, marker='.', alpha=0.2)


grid = onp.meshgrid(onp .linspace(-4, 4), onp.linspace(-4, 4))
grid = onp.stack(grid).T.reshape(-1, 2)

axes[1].scatter(*grid.T, c=np.clip(parallel_log_prob(params, grid, None).sum(0), -40)); plt.show()

plt.show()