import jax
import jax.numpy as np
import flax.linen as nn
from typing import Any


class ActNorm(nn.Module):
    def setup(self):
        init_inputs = self.param("kernel")  # idk if this can work
        self.log_weight = np.log(1.0 / (init_inputs.std(0) + 1e-6))
        self.bias = init_inputs.mean(0)

    def __call__(self, inputs, *args: Any, **kwds: Any):
        outputs = (inputs - self.bias) * np.exp(self.log_weight)
        log_det_jacobian = np.full(inputs.shape[:1], self.log_weight.sum())
        return outputs, log_det_jacobian

    def forward(self, inputs, *args: Any, **kwds: Any):
        return self(inputs, *args, **kwds)

    def inverse(self, inputs, *args: Any, **kwds: Any):
        outputs = inputs * np.exp(-self.log_weight) + self.bias
        log_det_jacobian = np.full(inputs.shape[:1], -self.log_weight.sum())
        return outputs, log_det_jacobian
