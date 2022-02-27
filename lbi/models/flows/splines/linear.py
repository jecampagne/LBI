import jax



def unconstrained_linear_spline(
    inputs, unnormalized_pdf, inverse=False, tail_bound=6.0, tails="linear"
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = jax.numpy.zeros_like(inputs)
    logabsdet = jax.numpy.zeros_like(inputs)

    if tails != "linear":
        raise RuntimeError("{} tails are not implemented.".format(tails))

    # if jax.numpy.any(inside_interval_mask):
    transformed_outputs, transformed_logabsdet = linear_spline(
        inputs=inputs, #[inside_interval_mask],
        unnormalized_pdf=unnormalized_pdf, #[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
    )

    outputs = inputs * outside_interval_mask + transformed_outputs * inside_interval_mask
    logabsdet = 0.0 * outside_interval_mask + transformed_logabsdet * inside_interval_mask


    return outputs, logabsdet



def linear_spline(
    inputs, unnormalized_pdf, inverse=False, left=0.0, right=1.0, bottom=0.0, top=1.0
):
    """
    Compute the piecewise linear function.

    References:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    > Adapted from https://github.com/bayesiains/nflows/blob/master/nflows/transforms/splines/linear.py
    """
    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_pdf.shape[-1]

    pdf = jax.nn.softmax(unnormalized_pdf, axis=-1)

    cdf = jax.numpy.cumsum(pdf, axis=-1)
    pad_width = [(0, 0) for i in range(len(pdf.shape) - 1)] + [(1, 0)]
    cdf = jax.numpy.pad(cdf, pad_width=pad_width, mode="constant", constant_values=0.0)

    if inverse:
        inv_bin_idx = jax.numpy.searchsorted(cdf, inputs, side="right")

        bin_boundaries = jax.numpy.linspace(0.0, 1.0, num_bins + 1).reshape(
            1 * len(inputs.shape), -1
        )
        bin_boundaries = jax.numpy.broadcast_to(
            bin_boundaries, (*inputs.shape, num_bins)
        )

        slopes = (cdf[..., 1:] - cdf[..., :-1]) / (
            bin_boundaries[..., 1:] - bin_boundaries[..., :-1]
        )
        offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

        inv_bin_idx = inv_bin_idx[:, None]
        input_slopes = jax.numpy.take_along_axis(slopes, inv_bin_idx, axis=-1)[..., 0]
        input_offsets = jax.numpy.take_along_axis(offsets, inv_bin_idx, axis=-1)[..., 0]

        outputs = (inputs - input_offsets) / input_slopes
        outputs = jax.numpy.clip(outputs, 0, 1)

        logabsdet = -jax.numpy.log(input_slopes)
    else:
        bin_pos = inputs * num_bins
        bin_pos = jax.numpy.clip(bin_pos, 0, num_bins - 1)
        bin_idx = jax.numpy.floor(bin_pos)

        alpha = bin_pos - bin_idx

        input_pdfs = jax.numpy.take_along_axis(pdf, bin_idx[..., None], axis=-1)[..., 0]

        outputs = jax.numpy.take_along_axis(cdf, bin_idx[..., None], axis=-1)[..., 0]
        outputs += alpha * input_pdfs
        outputs = jax.numpy.clip(outputs, 0, 1)

        bin_width = 1.0 / num_bins
        logabsdet = jax.numpy.log(input_pdfs) - jax.numpy.log(bin_width)

    if inverse:
        outputs = outputs * (right - left) + left
    else:
        outputs = outputs * (top - bottom) + bottom

    return outputs, logabsdet
