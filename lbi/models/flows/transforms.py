import jax.numpy as np
from lbi.models.flows.autoregressive import AutoregressiveTransform
from lbi.models.flows.splines.linear import unconstrained_linear_spline
from lbi.models.flows.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)


class MaskedLinearAutoregressiveTransform(AutoregressiveTransform):
    input_dim: int
    hidden_dim: int = 64
    context_dim: int = 0
    num_bins: int = 1
    tail_bound: float = 6.0
    act: str = "celu"

    def setup(self):
        output_dim_multiplier = 2
        super().setup(output_dim_multiplier=output_dim_multiplier)

    def linear_fn(self, inputs, params, inverse=False):
        log_weight, bias = params.split(2, axis=1)
        
        if inverse:
            outputs = inputs * np.exp(log_weight) + bias
            log_det_jacobian = -log_weight
        else:
            outputs = (inputs - bias) * np.exp(-log_weight)
            log_det_jacobian = -log_weight
            
        return outputs, log_det_jacobian

    def transform(self, inputs, context=None, inverse=False):
        transform_params = self.autoregressive_net(inputs, context=context)

        outputs, log_det_jacobian = self.linear_fn(
            inputs,
            transform_params,
            inverse=inverse,
        )
        return outputs, log_det_jacobian


class MaskedPiecewiseLinearAutoregressiveTransform(AutoregressiveTransform):
    input_dim: int
    hidden_dim: int = 64
    context_dim: int = 0
    num_bins: int = 4 
    tail_bound: float = 6.0
    act: str = "celu"

    def setup(self):
        output_dim_multiplier = self.num_bins
        super().setup(output_dim_multiplier=output_dim_multiplier)

    def transform(self, inputs, context=None, inverse=False):
        transform_params = self.autoregressive_net(inputs, context=context)
        batch_size = inputs.shape[0]

        unnormalized_pdf = transform_params.reshape(
            batch_size, -1, self.num_bins
        )
        outputs, log_det_jacobian = unconstrained_linear_spline(
            inputs,
            unnormalized_pdf,
            inverse=inverse,
            tail_bound=self.tail_bound,
        )
        return outputs, log_det_jacobian.sum(-1)


class MaskedPiecewiseRationalQuadraticAutoregressiveTransform(AutoregressiveTransform):
    input_dim: int
    hidden_dim: int = 64
    context_dim: int = 0
    num_bins: int = 4  
    tail_bound: float = 6.0
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3
    act: str = "celu"

    def setup(self):
        output_dim_multiplier = 3 * self.num_bins - 1
        super().setup(output_dim_multiplier=output_dim_multiplier)

    def transform(self, inputs, context=None, inverse=False):
        batch_size = inputs.shape[0]

        transform_params = self.autoregressive_net(inputs, context=context)
        transform_params = transform_params.reshape(batch_size, self.input_dim, -1)

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        unnormalized_widths /= np.sqrt(self.hidden_dim)
        unnormalized_heights /= np.sqrt(self.hidden_dim)

        outputs, log_det_jacobian = unconstrained_rational_quadratic_spline(
            inputs,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=inverse,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        
        return outputs, log_det_jacobian.sum(-1)


if __name__ == "__main__":
    import jax

    batch_size = 1024
    input_dim = 30
    hidden_dim = 64
    context_dim = 128
    num_bins = 50
    tail_bound = 2.0

    rng = jax.random.PRNGKey(0)

    x = jax.numpy.ones((1, input_dim))
    context = jax.numpy.ones((1, context_dim))

    model = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        num_bins=num_bins,
        tail_bound=tail_bound,
    )

    # test forward
    variables = model.init(rng, x, context)
    
    
    
    rand_context = jax.random.uniform(rng, (batch_size, context_dim))
    rand_batch = jax.random.uniform(rng, (batch_size, input_dim))
    rand_y = model.apply(variables, rand_batch, rand_context)

    import IPython; IPython.embed()
