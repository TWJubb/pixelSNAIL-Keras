import numpy as np
import tensorflow as tf


def get_causal_mask(canvas_size):
    """
    Create a basic mask for causality in the multi-head attention block; the mask (when multiplied by the attention
    tensor) sets to zero all elements in the "future". The attention tensor is (HW, HW) shape and element (i,j)
    describes how spatial location i influences spatial location j; therefore the mask will be

    1 0 0 0 ... 0
    1 1 0 0 ... 0
    1 1 1 0 ... 0
    1 1 1 1 ... 0
    : : : :     :
    0 0 0 0 ... 1

    canvas size is H.W (the total number of spatial positions in the attention map); e.g. for a 32x32 image HW=1024
    """
    causal_mask = np.zeros([canvas_size, canvas_size], dtype=np.float32)
    for i in range(canvas_size):
        causal_mask[i, :i] = 1.
    causal_mask = tf.constant(causal_mask, dtype=tf.float32)
    causal_mask = tf.reshape(causal_mask, [1, canvas_size, -1])
    return causal_mask
