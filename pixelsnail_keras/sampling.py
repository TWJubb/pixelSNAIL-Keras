import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import dateutil.tz
import os


def sample_from_discretized_mix_logistic(l, nr_mix):
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


class SampleCallback(tf.keras.callbacks.Callback):

    def __init__(self, save_every=10):

        self.train_loss = []
        self.test_loss = []
        self.save_every = save_every

        timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
        logdir = 'pixelsnail_data_%s' % (timestamp)

        self.save_dir = os.path.join(".", logdir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.training_file_path = self.save_dir + os.sep + "training.txt"
        if not os.path.exists(self.training_file_path):
            with open(self.training_file_path, "w") as f:
                f.write("")
        self.testing_file_path = self.save_dir + os.sep + "testing.txt"
        if not os.path.exists(self.testing_file_path):
            with open(self.testing_file_path, "w") as f:
                f.write("")

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.save_every == 0 and epoch != 0:
            samp = sample_from_model(self.model, shape=(32, 32, 3), batch_size=16)

            fig, ax = plt.subplots(4, 4, figsize=(10, 10))
            for i in range(4):
                for j in range(4):
                    img = samp[4 * i + j, :, :, :] * 127.5 + 127.5
                    ax[i, j].imshow(img.astype(int))
                    ax[i, j].axis('off')
            plt.subplots_adjust(hspace=0, wspace=0)
            fig.savefig(self.save_dir + os.sep + "samples_epoch_{}.png".format(epoch))

        with open(self.training_file_path, "a") as f:
            f.write("{}".format(epoch) + ",")
            f.write(",".join([str(t) for t in self.train_loss]))
            f.write("\n")
        with open(self.testing_file_path, "a") as f:
            f.write("{}".format(epoch) + ",")
            f.write(",".join([str(t) for t in self.test_loss]))
            f.write("\n")

        self.train_loss = []
        self.test_loss = []

    def on_train_batch_end(self, batch, logs=None):
        self.train_loss += [logs.get('loss')]

    def on_test_batch_end(self, batch, logs=None):
        self.test_loss += [logs.get('loss')]


def sample_from_model(model, shape=(32, 32, 3), batch_size=25):
    """
    Given a Keras model

    Parameters
    ----------

        model :
            A trained Keras pixelCNN model

        shape : tuple (H,W,C)
            The shape of a single image input to the model.

        batch_size, int
            The number of samples to geenrate in parallel.

    Returns
    -------

        numpy array of the sample, (batch_size, shape)

    """

    x_gen = np.zeros((batch_size,) + shape, dtype=np.float32)
    print("\n")
    for yi in range(shape[0]):
        print("Sampling batch of images : {:.1f} %".format(100 * yi / shape[0]), end="\r")
        for xi in range(shape[1]):
            new_x_gen = model.predict(x_gen)
            new_x_gen = sample_from_discretized_mix_logistic(new_x_gen, 10)
            x_gen[:, yi, xi, :] = new_x_gen[:, yi, xi, :]
    print("\n")
    return x_gen


def sample_from_model_and_plot(model, shape=(32, 32, 3), plot_grid=5):
    """
    Sample new images from a pixelCNN model and plot onto a grid

    Parameters
    ----------

        model: Keras Model

        shape: tuple (H,W,C) the image shape

        plot_grid: tuple (n_cols, n_rows) or int; determines the size of the grid of images to produce

    Rerturns
    --------

        fig, ax, sample : The matplotlib figure, Axes and the sample as a numpy array.
    """
    if type(plot_grid) == list:
        batch_size = plot_grid[0] * plot_grid[1]
    else:
        batch_size = plot_grid ** 2
        plot_grid = [plot_grid, plot_grid]

    sample = sample_from_model(model, shape=shape, batch_size=batch_size)

    fig, ax = plt.subplots(plot_grid[0], plot_grid[1], figsize=(10 * int(plot_grid[1] / plot_grid[0]), 10))
    for i in range(plot_grid[0]):
        for j in range(plot_grid[1]):
            img = sample[plot_grid[0] * i + j, :, :, :] * 127.5 + 127.5
            ax[i, j].imshow(img.astype(int))
            ax[i, j].axis('off')
    plt.subplots_adjust(hspace=0, wspace=0)

    return fig, ax, sample


def sample_from_model_occluded(image, model, nrows=16, ncols=8):
    """
    Block out a portion of an image and let the pixelCNN model fill in the rest.

    Parameters
    ----------

    :param image: image with shape (H,W,C)
    :param model:
    :param nrows: Number of rows to include
    :param ncols: Number of cols to include on the last non-occluded row
    :return:
    """
    shape = image.shape

    # block out some of the image
    x_reconstructed = np.zeros((1,) + shape, dtype=np.float32)
    x_reconstructed[:, :nrows, :, :] = image[:nrows, :, :]
    x_reconstructed[:, nrows, :ncols, :] = image[nrows, :ncols, :]

    # the blocked image
    occluded = x_reconstructed.copy()

    for yi in range(shape[0]):
        print("Sampling image : {:.1f} %".format(100 * yi / shape[0]), end="\r")
        if yi < nrows:
            continue
        for xi in range(shape[1]):
            if yi == nrows and xi < ncols:
                continue
            new_x_gen = model.predict(x_reconstructed)
            new_x_gen = sample_from_discretized_mix_logistic(new_x_gen, 10)
            x_reconstructed[:, yi, xi, :] = new_x_gen[:, yi, xi, :]
    return occluded, x_reconstructed
