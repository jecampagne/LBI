import jax
import flax.linen as nn
from typing import Sequence


class Flow(nn.Module):
    transformation: Sequence[nn.Module]
    prior: nn.Module

    def __call__(self, x, context=None):
        u, log_det_jacobian = self.transformation(x, context=context)
        base_log_prob = self.prior.log_prob(u)
        return u, base_log_prob + log_det_jacobian

    def forward(self, x, context=None):
        return self(x, context=context)

    def inverse(self, u, context=None):
        x, log_det_jacobian = self.transformation.inverse(u, context=context)
        base_log_prob = self.prior.log_prob(u)
        return x, base_log_prob + log_det_jacobian

    def log_pdf(self, x, context=None):
        return self(x, context=context)[1]

    def sample(self, rng, num_samples=0, context=None):
        prior_samples = self.prior.sample(rng, num_samples)
        x, log_det_jacobian = self.transformation.inverse(
            prior_samples, context=context
        )
        return x
    
    def sample_with_log_prob(self, rng, num_samples=0, context=None):
        prior_samples, prior_log_prob = self.prior.sample_with_log_prob(
            rng, num_samples
        )
        x, log_det_jacobian = self.transformation.inverse(
            prior_samples, context=context
        )
        return x, prior_log_prob + log_det_jacobian
