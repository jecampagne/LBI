import numpy as np
import jax
import jax.numpy as np
from lbi.models.flows.splines.utils import searchsorted

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = np.zeros_like(inputs)
    logabsdet = np.zeros_like(inputs)

    if tails != "linear":
        raise RuntimeError("{} tails are not implemented.".format(tails))

    pad_width = [(0, 0) for i in range(len(unnormalized_derivatives.shape) - 1)] + [
        (1, 1)
    ]
    unnormalized_derivatives = np.pad(
        unnormalized_derivatives,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0,
    )

    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives = unnormalized_derivatives.at[..., 0].set(constant)
    unnormalized_derivatives = unnormalized_derivatives.at[..., -1].set(constant)

    transformed_outputs, transformed_logabsdet = rational_quadratic_spline(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    outputs = (
        inputs * outside_interval_mask + transformed_outputs * inside_interval_mask
    )
    logabsdet = (
        0.0 * outside_interval_mask + transformed_logabsdet * inside_interval_mask
    )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = jax.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = np.cumsum(widths, axis=-1)

    pad_width = [(0, 0) for i in range(len(cumwidths.shape) - 1)] + [(1, 0)]
    cumwidths = np.pad(
        cumwidths,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0,
    )

    cumwidths = (right - left) * cumwidths + left
    cumwidths = cumwidths.at[..., 0].set(left)
    cumwidths = cumwidths.at[..., -1].set(right)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + jax.nn.softplus(unnormalized_derivatives)

    heights = jax.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = np.cumsum(heights, axis=-1)

    pad_width = [(0, 0) for i in range(len(cumheights.shape) - 1)] + [(1, 0)]
    cumheights = np.pad(
        cumheights,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0,
    )

    cumheights = (top - bottom) * cumheights + bottom
    cumheights = cumheights.at[..., 0].set(bottom)
    cumheights = cumheights.at[..., -1].set(top)
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]


    input_cumwidths = np.take_along_axis(cumwidths, bin_idx, axis=-1)[..., 0]
    input_bin_widths = np.take_along_axis(widths, bin_idx, axis=-1)[..., 0]

    input_cumheights = np.take_along_axis(cumheights, bin_idx, axis=-1)[..., 0]
    delta = heights / widths
    input_delta = np.take_along_axis(delta, bin_idx, axis=-1)[..., 0]

    input_derivatives = np.take_along_axis(derivatives, bin_idx, axis=-1)[..., 0]
    input_derivatives_plus_one = np.take_along_axis(
        derivatives[..., 1:], bin_idx, axis=-1
    )[..., 0]

    input_heights = np.take_along_axis(heights, bin_idx, axis=-1)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = np.power(b, 2) - 4 * a * c
        # assert np.all(discriminant >= 0)

        root = (2 * c) / (-b - np.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = np.power(input_delta, 2) * (
            input_derivatives_plus_one * np.power(root, 2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * np.power((1 - root), 2)
        )
        logabsdet = np.log(derivative_numerator) - 2 * np.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * np.power(theta, 2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = np.power(input_delta, 2) * (
            input_derivatives_plus_one * np.power(theta, 2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * np.power((1 - theta), 2)
        )
        logabsdet = np.log(derivative_numerator) - 2 * np.log(denominator)

        return outputs, logabsdet
