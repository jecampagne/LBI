import jax.numpy as np
from lbi.models.flows.made import MaskedTransform
import lbi.models.flows.splines.utils as utils
from flax.linen.module import compact
import flax.linen as nn


class AutoregressiveTransform(nn.Module):
    input_dim: int
    hidden_dim: int = 64
    context_dim: int = 0
    act: str = "celu"
    precision: int = 4

    def setup(self, output_dim_multiplier):
        self.autoregressive_net = MaskedTransform(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            num_hidden=1,
            output_dim_multiplier=output_dim_multiplier,
            act=self.act,
        )

    @compact
    def __call__(self, inputs, context=None):
        outputs, logabsdet = self.forward(inputs, context=context)
        return outputs, utils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        autoregressive_params = self.autoregressive_net(inputs, context=context)
        outputs, logabsdet = self._elementwise_transform(
            inputs, autoregressive_params, inverse=False
        )
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        outputs = np.zeros_like(inputs)
        
        for i_col in range(inputs.shape[1]):
            autoregressive_params = self.autoregressive_net(outputs, context=context)
            intermediate_outputs, logabsdet = self._elementwise_transform(
                inputs, autoregressive_params, inverse=True,
            )
            outputs = outputs.at[:, i_col].set(intermediate_outputs[:, i_col])

        return outputs, -utils.sum_except_batch(logabsdet)

    def _elementwise_transform(self, inputs, context=None, inverse=False):
        raise NotImplementedError
