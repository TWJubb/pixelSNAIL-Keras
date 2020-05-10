import numpy as np
import os
import tensorflow as tf

from pixelsnail_keras.wn import WeightNormalization
from tensorflow.keras.layers import Layer, ZeroPadding2D, Conv2D, Dropout, Dense, Input, Concatenate, Reshape, \
    Cropping2D, Activation

from pixelsnail_keras.utils import get_causal_mask


# ----------------------------------------------------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------------------------------------------------

class Shift(Layer):
    """
    A layer to shift a tensor
    """

    def __init__(self, direction, size=1, **kwargs):
        self.size = size
        self.direction = direction
        super(Shift, self).__init__(**kwargs)

        if self.direction == "down":
            self.pad = ZeroPadding2D(padding=((self.size, 0), (0, 0)), data_format="channels_last")
            self.crop = Cropping2D(((0, self.size), (0, 0)))
        elif self.direction == "right":
            self.pad = ZeroPadding2D(padding=((0, 0), (self.size, 0)), data_format="channels_last")
            self.crop = Cropping2D(((0, 0), (0, self.size)))

    def build(self, input_shape):
        super(Shift, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.crop(self.pad(x))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(Shift, self).get_config()
        config.update({'direction': self.direction,
                       'size': self.size})
        return config


class CausalConv2D(Layer):
    """
    Basic causal convolution layer; implementing causality and weight normalization.
    """

    def __init__(self, filters, kernel_size=[3, 3], weight_norm=True, shift=None, strides=1, activation="relu",
                 **kwargs):
        self.output_dim = filters
        super(CausalConv2D, self).__init__(**kwargs)

        pad_h = ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2)
        pad_v = ((kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2)
        if shift == "down":
            pad_h = ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2)
            pad_v = (kernel_size[0] - 1, 0)
        elif shift == "right":
            pad_h = (kernel_size[1] - 1, 0)
            pad_v = ((kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2)
        elif shift == "downright":
            pad_h = (kernel_size[1] - 1, 0)
            pad_v = (kernel_size[0] - 1, 0)

        self.padding = (pad_v, pad_h)

        self.pad = ZeroPadding2D(padding=self.padding,
                                 data_format="channels_last")

        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, padding="VALID", strides=strides,
                           activation=activation)

        if weight_norm:
            self.conv = WeightNormalization(self.conv, data_init=True)

    def build(self, input_shape):
        super(CausalConv2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.conv(self.pad(x))

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super(CausalConv2D, self).get_config()
        config.update({'padding': self.padding,
                       'output_dim': self.output_dim})
        return config


class NetworkInNetwork(Layer):
    """
    Basic causal convolution layer; implementing causality and weight normalization.

    """

    def __init__(self, filters, activation=None, weight_norm=True, **kwargs):
        self.filters = filters
        self.activation = activation
        super(NetworkInNetwork, self).__init__(**kwargs)

        if weight_norm:
            self.dense = WeightNormalization(Dense(self.filters))
        else:
            self.dense = Dense(self.filters)
        self.activation = Activation(self.activation)

    def build(self, input_shape):
        super(NetworkInNetwork, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = self.dense(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters)

    def get_config(self):
        config = super(NetworkInNetwork, self).get_config()
        return config


class CausalAttention(Layer):
    """
    """

    def __init__(self, **kwargs):
        super(CausalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CausalAttention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        key, query, value = x

        nr_chns = key.shape[-1]
        mixin_chns = value.shape[-1]

        canvas_size = int(np.prod(key.shape[1:-1]))
        canvas_size_q = int(np.prod(query.shape[1:-1]))
        causal_mask = get_causal_mask(canvas_size_q)

        q_m = Reshape((canvas_size_q, nr_chns))(tf.debugging.check_numerics(query, "badQ"))
        k_m = Reshape((canvas_size, nr_chns))(tf.debugging.check_numerics(key, "badK"))
        v_m = Reshape((canvas_size, mixin_chns))(tf.debugging.check_numerics(value, "badV"))

        dot = tf.matmul(q_m, k_m, transpose_b=True)
        dk = tf.cast(nr_chns, tf.float32)
        causal_probs = tf.nn.softmax(dot / tf.math.sqrt(dk) - 1e9 * causal_mask, axis=-1) * causal_mask
        # causal_probs = tf.nn.softmax(dot, axis=-1) * causal_mask
        mixed = tf.matmul(causal_probs, v_m)

        # mixed = tf.keras.layers.Attention(causal=True)([q_m, v_m, k_m])
        out = Reshape(query.shape[1:-1] + [mixin_chns])(mixed)
        out = tf.debugging.check_numerics(out, "bad mixed")

        return out

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super(CausalAttention, self).get_config()
        return config


# ----------------------------------------------------------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------------------------------------------------------


def GatedResidualBlock(x, aux=None,
                       nonlinearity=None,
                       dropout=0.0,
                       conv1=None,
                       conv2=None):
    """
    x, aux are both logits; logits are also returned from the function
    """
    filters = x.shape[-1]
    activation = Activation(nonlinearity)

    # should this not have activation??
    c1 = conv1(activation(x))

    if aux is not None:
        # add short-cut connection if auxiliary input 'a' is given
        # using NIN (network-in-network)
        c1 += NetworkInNetwork(filters, activation=None)(activation(aux))

    # c1 is passed through a non-linearity step here; not sure if it is needed??
    c1 = activation(c1)

    if dropout > 0.0:
        c1 = Dropout(dropout)(c1)

    c2 = conv2(c1)

    # Gating ; split into two pieces along teh channels
    a, b = tf.split(c2, 2, 3)
    c3 = a * tf.nn.sigmoid(b)

    # skip connection to input
    x_out = x + c3

    return x_out


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------

def pixelSNAIL(attention=True, out_channels=None, num_pixel_blocks=1, num_grb_per_pixel_block=1, dropout=0.0,
               nr_filters=128):
    nr_logistic_mix = 10
    kernel_size = 3

    x_in = Input(shape=(32, 32, 3))

    k_d = [kernel_size - 1, kernel_size]
    k_dr = [kernel_size - 1, kernel_size - 1]

    u = Shift("down")(CausalConv2D(nr_filters, [kernel_size - 1, kernel_size], shift="down")(x_in))
    ul = Shift("down")(CausalConv2D(nr_filters, [1, kernel_size], shift="down")(x_in))
    ul += Shift("right")(CausalConv2D(nr_filters, [kernel_size - 1, 1], shift="downright")(x_in))

    for i in range(num_pixel_blocks):
        for j in range(num_grb_per_pixel_block):
            u = GatedResidualBlock(x=u, aux=None,
                                   nonlinearity="elu",
                                   dropout=dropout,
                                   conv1=CausalConv2D(filters=nr_filters, kernel_size=k_d, shift="down",
                                                      activation="elu", name="causalconv_u_1_{}_{}".format(i, j)),
                                   conv2=CausalConv2D(filters=2 * nr_filters, kernel_size=k_d, shift="down",
                                                      activation="elu", name="causalconv_u_2_{}_{}".format(i, j)))
            ul = GatedResidualBlock(x=ul, aux=u,
                                    nonlinearity="elu",
                                    dropout=dropout,
                                    conv1=CausalConv2D(filters=nr_filters, kernel_size=k_dr, shift="downright",
                                                       activation="elu", name="causalconv_ul_1_{}_{}".format(i, j)),
                                    conv2=CausalConv2D(filters=2 * nr_filters, kernel_size=k_dr, shift="downright",
                                                       activation="elu", name="causalconv_ul_2_{}_{}".format(i, j)))

        if attention:
            content = Concatenate(axis=3)([x_in, ul])

            content = tf.debugging.check_numerics(content, "bad conent")
            channels = content.shape[-1]
            kv = GatedResidualBlock(x=content, aux=None,
                                    nonlinearity="elu",
                                    dropout=dropout,
                                    conv1=NetworkInNetwork(filters=channels, activation=None),
                                    conv2=NetworkInNetwork(filters=2 * channels, activation=None))
            kv = NetworkInNetwork(filters=2 * nr_filters, activation=None)(kv)
            key, value = tf.split(kv, 2, axis=3)

            query = GatedResidualBlock(x=ul, aux=None,
                                       nonlinearity="elu",
                                       dropout=dropout,
                                       conv1=NetworkInNetwork(filters=nr_filters, activation=None),
                                       conv2=NetworkInNetwork(filters=2 * nr_filters, activation=None))
            query = NetworkInNetwork(filters=nr_filters, activation=None)(query)
            a = CausalAttention()([key, query, value])
            a = tf.debugging.check_numerics(a, "bad a!!")
        else:
            a = None

        ul = GatedResidualBlock(x=ul, aux=a,
                                nonlinearity="elu",
                                dropout=dropout,
                                conv1=NetworkInNetwork(filters=nr_filters, activation=None),
                                conv2=NetworkInNetwork(filters=2 * nr_filters, activation=None))

    ul = Activation("elu")(ul)

    if out_channels is not None:
        filters = out_channels
        x_out = NetworkInNetwork(filters=filters, activation=None)(ul)
    else:
        filters = 10 * nr_logistic_mix
        x_out = NetworkInNetwork(filters=filters, activation=None)(ul)

    model = tf.keras.Model(inputs=x_in, outputs=x_out)

    return model
