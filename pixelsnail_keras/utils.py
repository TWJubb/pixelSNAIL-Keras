import numpy as np
import tensorflow as tf

def get_causal_mask(canvas_size, rate=1):
    """

    :param canvas_size:
    :param rate:
    :return:
    """
    causal_mask = np.zeros([canvas_size, canvas_size], dtype=np.float32)
    for i in range(canvas_size):
        causal_mask[i, :i] = 1.
    causal_mask = tf.constant(causal_mask, dtype=tf.float32)

    if rate > 1:
        dim = int(np.sqrt(canvas_size))
        causal_mask = tf.reshape(causal_mask, [canvas_size, dim, dim, 1])
        causal_mask = -tf.nn.max_pool(-causal_mask, [1, rate, rate, 1], [1, rate, rate, 1], 'SAME')

    causal_mask = tf.reshape(causal_mask, [1, canvas_size, -1])
    return causal_mask