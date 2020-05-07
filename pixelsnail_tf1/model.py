"""
The core Pixel-CNN model
"""
import functools
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import nn as nn

from tensorflow.python.ops import variables
def passthrough(obj, value):
    return value
variables.Variable._build_initializer_expr=passthrough

def model_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10,
               resnet_nonlinearity='concat_elu', attention=False, nr_attn_block=1):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema,
                   dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise ('resnet nonlinearity ' +
                   resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                [
                    ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                    ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                ],
                axis=3
            )
            # add channel of ones to distinguish image from padding later on
            # stream for pixels above
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            u_list = [nn.down_shift(nn.down_shifted_conv2d(
                x_pad, num_filters=nr_filters, filter_size=[2, 3]))]
            # stream for up and to the left
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,
                                                                                                               1]))]

            for attn_rep in range(nr_attn_block):
                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

                if attention:
                    ul = ul_list[-1]
                    raw_content = tf.concat([x, ul, background], axis=3)
                    key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2), 2, axis=3)
                    query = nn.nin(nn.gated_resnet(tf.concat([ul, background], axis=3), conv=nn.nin), nr_filters)
                    mixed = nn.causal_attention(key, mixin, query)

                    ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))

            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)
            return x_out
