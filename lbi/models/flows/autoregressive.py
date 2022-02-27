import jax.numpy as np
from lbi.models.flows.made import MaskedTransform
from flax.linen.module import compact
import flax.linen as nn


class AutoregressiveTransform(nn.Module):
    input_dim: int
    hidden_dim: int = 64
    context_dim: int = 0
    output_dim_multiplier: int = 4  # number of bins
    act: str = "celu"

    def setup(self):
        self.autoregressive_net = MaskedTransform(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            num_hidden=1,
            output_dim_multiplier=self.output_dim_multiplier,
            act=self.act,
        )
        
    @compact
    def __call__(self, inputs, context=None):
        return self.transform(
            inputs, context=context, inverse=False
        )

    def transform(self, inputs, context=None, inverse=False):
        raise NotImplementedError

    def forward(self, inputs, context=None):
        return self(inputs, context=context)

    def inverse(self, inputs, context=None):
        outputs = np.zeros_like(inputs)
        for i_col in range(inputs.shape[1]):
            interemediate_outputs, log_det_jacobian = self.transform(
                inputs, context=None, inverse=True
            )
            outputs = outputs.at[:, i_col].set(interemediate_outputs[:, i_col])

        log_det_jacobian = -log_det_jacobian.sum(-1)
        return outputs, log_det_jacobian
