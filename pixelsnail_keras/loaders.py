from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.datasets import cifar10


def load_cifar_data(preprocess, batch_size=32):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    cap = x_train.shape[0] // batch_size
    x_train = x_train[:cap * batch_size, ...]

    cap = x_test.shape[0] // batch_size
    x_test = x_test[:cap * batch_size, ...]

    if preprocess is not None:
        x_train = preprocess(x_train)
        x_test = preprocess(x_test)

    datagen = ImageDataGenerator()
    datagen_train = datagen.flow(x_train.astype(float), x_train.astype(float),
                                 batch_size=batch_size)
    datagen_test = datagen.flow(x_test.astype(float), x_test.astype(float),
                                batch_size=batch_size)

    return datagen_train, datagen_test


def load_gemstone_data(preprocess, batch_size=32):
    datagen = ImageDataGenerator(preprocessing_function=preprocess)

    datagen_train = datagen.flow_from_directory("/home/tom/datasets/gemstones/train/",
                                                target_size=(32, 32),
                                                batch_size=batch_size,
                                                class_mode=None,
                                                shuffle=True)
    datagen_test = datagen.flow_from_directory("/home/tom/datasets/gemstones/test/",
                                               target_size=(32, 32),
                                               batch_size=batch_size,
                                               class_mode=None,
                                               shuffle=True)
    x_train = np.zeros((0, 32, 32, 3))
    for data in datagen_train:
        if datagen_train.total_batches_seen > len(datagen_train):
            break
        x_train = np.concatenate([x_train, data])
    print(x_train.max(), x_train.min())

    x_test = np.zeros((0, 32, 32, 3))
    for data in datagen_test:
        if datagen_test.total_batches_seen > len(datagen_test):
            break
        x_test = np.concatenate([x_test, data])

    cap = x_train.shape[0] // batch_size
    x_train = x_train[:cap * batch_size, ...]

    cap = x_test.shape[0] // batch_size
    x_test = x_test[:cap * batch_size, ...]

    datagen = ImageDataGenerator()
    datagen_train = datagen.flow(x_train.astype(float), x_train.astype(float),
                                 batch_size=batch_size)
    datagen_test = datagen.flow(x_test.astype(float), x_test.astype(float),
                                batch_size=batch_size)

    return datagen_train, datagen_test
