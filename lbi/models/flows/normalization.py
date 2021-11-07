import operator
from flax.linen.module import compact
import jax
import jax.numpy as np
import flax.linen as nn
from typing import Any


# class ActNorm(nn.Module):
#     input_dim: int 
#     log_weight: None = None
#     bias: None = None
    
#     def setup(self):
#         log_weight = np.zeros(self.input_dim)
#         bias = np.zeros(self.input_dim)

#     def __call__(self, inputs, *args: Any, **kwds: Any):
#         outputs = (inputs - self.bias) * np.exp(self.log_weight)
#         log_det_jacobian = np.full(inputs.shape[:1], self.log_weight.sum())
#         return outputs, log_det_jacobian

#     def forward(self, inputs, *args: Any, **kwds: Any):
#         return self(inputs, *args, **kwds)

#     def inverse(self, inputs, *args: Any, **kwds: Any):
#         outputs = inputs * np.exp(-self.log_weight) + self.bias
#         log_det_jacobian = np.full(inputs.shape[:1], -self.log_weight.sum())
#         return outputs, log_det_jacobian

class ActNorm(nn.Module):
    scale: float = 1.0
    eps: float = 1e-8
    
    @compact
    def __call__(self, inputs, logdet=0, reverse=False):
        axes = tuple(i for i in range(len(inputs.shape) - 1))
        def dd_mean_initializer(key, shape):
            """Data-dependent init for mu"""
            nonlocal inputs
            x_mean = np.mean(inputs, axis=axes, keepdims=True)
            return - x_mean
        
        def dd_stddev_initializer(key, shape):
            """Data-dependent init for sigma"""
            nonlocal inputs
            x_var = np.mean(inputs**2, axis=axes, keepdims=True)
            var = self.scale / (np.sqrt(x_var) + self.eps)
            return var
        
        shape = (1,) *len(axes) + (inputs.shape[-1],)
        mu = self.param('actnorm_mean', dd_mean_initializer, shape)
        sigma = self.param('actnorm_stddev', dd_stddev_initializer, shape)
        
        logsigma = np.log(np.abs(sigma))
        log_det_jacobian = reduce(operator.mul, (inputs.shape[i] for i in range(1, len(inputs.shape) - 1)), 1)
        
        if reverse:
            outputs = inputs / (sigma + self.eps) - mu
            log_det_jacobian = - log_det_jacobian * np.sum(logsigma)
        else:
            outputs = sigma * (inputs + mu)
            log_det_jacobian = log_det_jacobian * np.sum(logsigma)
            
        return outputs, log_det_jacobian
    
    def forward(self, inputs, context=None):
        return self(inputs, context, reverse=False)
    
    def inverse(self, inputs, context=None):
        return self(inputs, context, reverse=True)
        