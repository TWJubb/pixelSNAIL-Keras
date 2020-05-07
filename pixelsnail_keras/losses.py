import numpy as np
import tensorflow as tf


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat([x, -x], axis))


def logsumexp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.math.log(tf.reduce_sum(tf.exp(x - m2), axis))


def discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = x.shape  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = l.shape  # predicted distribution, e.g. (B,32,32,100)

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs, means, log_scales, coeffs = tf.split(l,
                                                      num_or_size_splits=[nr_mix, 3 * nr_mix, 3 * nr_mix, 3 * nr_mix],
                                                      axis=-1)

    log_scales = tf.maximum(log_scales, -7.)
    log_scales = tf.concat(tf.split(tf.expand_dims(log_scales, -2), 3, -1), -2)
    coeffs = tf.nn.tanh(coeffs)

    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    m1, m2, m3 = tf.split(means, num_or_size_splits=3, axis=-1)
    c1, c2, c3 = tf.split(coeffs, num_or_size_splits=3, axis=-1)
    x1, x2, x3 = tf.split(x, num_or_size_splits=3, axis=-1)

    m2 += c1 * x1
    m3 += c2 * x1 + c3 * x2

    means = tf.concat([tf.expand_dims(m1, axis=-2),
                       tf.expand_dims(m2, axis=-2),
                       tf.expand_dims(m3, axis=-2)], axis=-2)

    x = tf.expand_dims(x, -1)
    x_c = tf.subtract(x, means)

    inv_stdv = tf.exp(-log_scales)

    plus_in = inv_stdv * (x_c + 1. / 255.)
    cdf_plus = tf.nn.sigmoid(plus_in)

    min_in = inv_stdv * (x_c - 1. / 255.)
    cdf_min = tf.nn.sigmoid(min_in)
    # log probability for edge case o
    # f 0 (before scaling)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * x_c
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min,
                                                            tf.where(cdf_delta > 1e-5,
                                                                     tf.math.log(tf.maximum(cdf_delta, 1e-12)),
                                                                     log_pdf_mid - np.log(127.5))))

    # sum log probs ==> multiply the probs
    log_probs = tf.reduce_sum(log_probs, 3)
    log_probs += tf.nn.log_softmax(logit_probs - tf.reduce_max(logit_probs, -1, keepdims=True))
    loss = -tf.reduce_sum(logsumexp(log_probs))

    n = tf.cast(tf.size(x), tf.float32)
    return tf.cast(loss, tf.float32) / (n * np.log(2))


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = list(l.shape)
    xs = ls[:-1] + [3]

    logit_probs, means, log_scales, coeffs = tf.split(l,
                                                      num_or_size_splits=[nr_mix, 3 * nr_mix, 3 * nr_mix, 3 * nr_mix],
                                                      axis=-1)
    means = tf.reshape(means, shape=xs + [nr_mix])
    log_scales = tf.reshape(log_scales, shape=xs + [nr_mix])
    coeffs = tf.reshape(coeffs, shape=xs + [nr_mix])

    # unpack parameters
    logit_probs = tf.nn.log_softmax(logit_probs - tf.reduce_max(logit_probs, -1,
                                                                keepdims=True))  # log_prob_from_logits(tf.convert_to_tensor(l[:, :, :, :nr_mix])).numpy()

    # sample mixture indicator from softmax
    sel = tf.argmax(logit_probs - tf.math.log(
        -tf.math.log(tf.random.uniform(list(logit_probs.shape), minval=1e-5, maxval=1. - 1e-5))), 3)
    sel = tf.one_hot(sel, depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1, nr_mix])

    # select logistic parameters
    means = tf.reduce_sum(means * sel, 4)
    log_scales = tf.maximum(tf.reduce_sum(log_scales * sel, 4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(coeffs) * sel, 4)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random.uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales) * (tf.math.log(u) - tf.math.log(1. - u))

    x0 = tf.minimum(tf.maximum(x[:, :, :, 0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)

    return tf.concat([tf.reshape(x0, xs[:-1] + [1]),
                      tf.reshape(x1, xs[:-1] + [1]),
                      tf.reshape(x2, xs[:-1] + [1])], 3)


def discretized_mix_logistic_loss_new(x, l):
    """
    Log-likelihood loss for mixture of discretized logistics in "bits per dimension" units.

    bits per dimension = NLL / (log(2) * B * H * W * C)

    with negative log-likelihood;

    $$
    NLL(x) = \sum_i pi_i * logistic(x, mu_i, s_i)
    $$

    For example, a batch of images of (B,H,W,C) (1,32,32,3) as for cifar-10; with a likelihood build from a mixture of
    nr_mix=10 logistic distributions expects an output from the network of (1,32,32,100) where the output feature length N=100
    is (nr_mix * (1 + 3 + 3 + 3)) corresponding to the pi_i (mixture indicator), mu_i, s_i and c_i.

    Parameters
    ----------

        x, Tensor (B,H,W,3) :
            The input RGB image, which must be scaled to the interval [-1,1]

        l, Tensor (B,H,W,N) :
            The output from a pixelCNN network with the same spatial size as the input, and where the output channels N
            is (nr_mix * (1 + 3 + 3 + 3)) corresponding to the pi_i (mixture indicator), mu_i, s_i and c_i.

    Returns
    -------

        loss, float
    """
    ls = l.shape
    # number of logistics in distribution
    nr_mix = int(ls[-1] / 10)

    # unpacking the params of the mixture of logistics
    split = [nr_mix, 3 * nr_mix, 3 * nr_mix, 3 * nr_mix]
    pi_i, mu_i, log_s_i, rgb_coefficients = tf.split(l, num_or_size_splits=split, axis=-1)

    log_s_i = tf.maximum(log_s_i, -7.)
    log_s_i = tf.concat(tf.split(tf.expand_dims(log_s_i, -2), 3, -1), -2)
    rgb_coefficients = tf.nn.tanh(rgb_coefficients)
    one_over_s_i = tf.exp(-log_s_i)

    # get mu_i and adjust based on preceding sub-pixels
    mu_r, mu_g, mu_b = tf.split(mu_i, num_or_size_splits=3, axis=-1)
    c0, c1, c2 = tf.split(rgb_coefficients, num_or_size_splits=3, axis=-1)
    x_r, x_g, x_b = tf.split(x, num_or_size_splits=3, axis=-1)

    mu_g += c0 * x_r
    mu_b += c1 * x_r + c2 * x_g

    mu_i = tf.concat([tf.expand_dims(mu_r, axis=-2),
                      tf.expand_dims(mu_g, axis=-2),
                      tf.expand_dims(mu_b, axis=-2)], axis=-2)

    x = tf.expand_dims(x, -1)
    x_minus_mu = tf.subtract(x, mu_i)

    # log probability for edge case 0
    plus_in = one_over_s_i * (x_minus_mu + 1. / 255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    min_in = one_over_s_i * (x_minus_mu - 1. / 255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min
    mid_in = one_over_s_i * x_minus_mu

    # log probability in the center of the bin, to be used in extreme cases
    log_pdf_mid = mid_in - log_s_i - 2. * tf.nn.softplus(mid_in)

    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min,
                                                            tf.where(cdf_delta > 1e-5,
                                                                     tf.math.log(tf.maximum(cdf_delta, 1e-12)),
                                                                     log_pdf_mid - np.log(127.5))))

    # sum log probs ==> multiply the probs
    log_probs = tf.reduce_sum(log_probs, 3)
    log_probs += tf.nn.log_softmax(pi_i - tf.reduce_max(pi_i, -1, keepdims=True))
    loss = -tf.reduce_sum(logsumexp(log_probs))

    n = tf.cast(tf.size(x), tf.float32)
    return tf.cast(loss, tf.float32) / (n * np.log(2))


def sample_from_discretized_mix_logistic_new(l, nr_mix):
    """
    Sampling function for the pixelCNN family of algorithms which will generate an RGB image.

    Parameters
    ----------

    l, Tensor (B,H,W,N)
        The output from a pixelCNN network, where N is (nr_mix * (1 + 3 + 3 + 3)) corresponding to the pi_i (mixture
        indicator), mu_i, s_i and c_i.

    nr_mix, int
        The number of logistic distributions included in the network output. Usually 5 or 10

    Returns
    -------

    Tensor, (B,H,W,3) : The RGB values of the sampled pixels


    """
    ls = list(l.shape)
    xs = ls[:-1] + [3]

    # split the network output into its pieces
    split = [nr_mix, 3 * nr_mix, 3 * nr_mix, 3 * nr_mix]
    logit_probs, means, log_s, coeff = tf.split(l, num_or_size_splits=split, axis=-1)
    means = tf.reshape(means, shape=xs + [nr_mix])
    scale = tf.exp(tf.reshape(log_s, shape=xs + [nr_mix]))
    coeff = tf.reshape(tf.nn.tanh(coeff), shape=xs + [nr_mix])

    # the probabilities for each "mixture indicator"
    logit_probs = tf.nn.log_softmax(logit_probs - tf.reduce_max(logit_probs, -1, keepdims=True))

    # sample "mixture indicator" from softmax using Gumbel-max trick
    rand_sample = -tf.math.log(tf.random.uniform(list(logit_probs.shape), minval=1e-5, maxval=1. - 1e-5))
    sel = tf.argmax(logit_probs - tf.math.log(rand_sample), 3)
    sel = tf.one_hot(sel, depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1, nr_mix])

    # select logistic parameters from the sampled mixture indicator
    means = tf.reduce_sum(means * sel, 4)
    scale = tf.maximum(tf.reduce_sum(scale * sel, 4), -7.)
    coeff = tf.reduce_sum(coeff * sel, 4)

    # sample the RGB values (before adding linear dependence)
    u = tf.random.uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    sample_mu = means + scale * (tf.math.log(u) - tf.math.log(1. - u))
    mu_hat_r, mu_hat_g, mu_hat_b = tf.split(sample_mu, num_or_size_splits=3, axis=-1)

    # include the linear dependence of r->g and r,g->b
    c0, c1, c2 = tf.split(coeff, num_or_size_splits=3, axis=-1)
    x_r = tf.clip_by_value(mu_hat_r, -1.0, 1.0)
    x_g = tf.clip_by_value(mu_hat_g + c0 * x_r, -1.0, 1.0)
    x_b = tf.clip_by_value(mu_hat_b + c1 * x_r + c2 * x_g, -1.0, 1.0)

    return tf.concat([tf.reshape(x_r, xs[:-1] + [1]),
                      tf.reshape(x_g, xs[:-1] + [1]),
                      tf.reshape(x_b, xs[:-1] + [1])], 3)

