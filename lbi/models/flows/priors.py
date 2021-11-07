import jax
from jax.scipy.stats import norm
from flax.linen import nn 


class Normal(nn.Module):
    """
    Returns:
        A function mapping ``(rng, input_dim)`` to a ``(params, log_pdf, sample)`` triplet.
    """
    dim: int
    
    def log_pdf(inputs):
        return norm.logpdf(inputs).sum(1)

    def sample(self, rng, num_samples=1):
        return jax.random.normal(rng, (num_samples, self.dim))

    def sample_with_log_prob(self, rng, num_samples=1):
        samples = self.sample(rng, num_samples)
        log_probs = self.log_pdf(samples)
        return samples, log_probs