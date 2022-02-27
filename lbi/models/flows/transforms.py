import jax.numpy as np
from lbi.models.flows.autoregressive import AutoregressiveTransform
from lbi.models.flows.splines.linear import linear_spline, unconstrained_linear_spline


class MaskedPiecewiseLinearAutoregressiveTransform(AutoregressiveTransform):
    input_dim: int
    hidden_dim: int = 64
    context_dim: int = 0
    output_dim_multiplier: int = 4  # number of bins
    tail_bound: float = 6.0
    act: str = "celu"

    def setup(self):
        super().setup()

    def transform(self, inputs, context=None, inverse=False):
        transform_params = self.autoregressive_net(inputs, context=context).split(
            self.output_dim_multiplier, axis=1
        )
        batch_size = inputs.shape[0]
        transform_params = np.stack(transform_params, axis=-1)
        
        unnormalized_pdf = transform_params.reshape(batch_size, -1, self.output_dim_multiplier)
        outputs, log_det_jacobian = unconstrained_linear_spline(
            inputs, unnormalized_pdf, inverse=inverse, tail_bound=self.tail_bound, 
        )
        return outputs, log_det_jacobian.sum(-1)




if __name__ == "__main__":
    import jax

    input_dim = 3
    hidden_dim = 64
    context_dim = 1
    output_dim_multiplier = 2

    rng = jax.random.PRNGKey(0)

    x = jax.numpy.ones((1, input_dim))
    context = jax.numpy.ones((1, context_dim))

    model = MaskedPiecewiseLinearAutoregressiveTransform(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        output_dim_multiplier=5,
    )

    # test forward
    variables = model.init(rng, x, context)
    y = model.apply(variables, x, context)

    from IPython import embed; embed()
