import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds



def load_data(DATA, BATCH_SIZE):
    if DATA == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_train = tf.where(x_train > 0.5, 1.0, 0.0)
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        x_test = tf.where(x_test > 0.5, 1.0, 0.0)
    elif DATA == 'fmnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    elif DATA == 'omniglot':
        train_ds = tfds.load(DATA, split='train', shuffle_files=True, as_supervised=False)
        x_train = list(map(lambda x: tf.squeeze(tf.image.resize(tf.image.rgb_to_grayscale(tf.cast(x['image'], dtype=tf.float32)), [28, 28])), train_ds))
        x_train = np.array(x_train)
        test_ds = tfds.load(DATA, split='test', shuffle_files=True, as_supervised=False)
        x_test = list(map(lambda x: tf.squeeze(tf.image.resize(tf.image.rgb_to_grayscale(tf.cast(x['image'], dtype=tf.float32)), [28, 28])), test_ds))
        x_test = np.array(x_test)
        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        x_train = tf.where(x_train > 0.5, 1.0, 0.0)
        x_test = tf.where(x_test > 0.5, 1.0, 0.0)
    else:
        raise NotImplementedError(f"Data {DATA} not implemented")
    
    TRAIN_BUF = 60000
    VAL_BUF = 1000
    TEST_BUF = 9000

    if DATA == 'omniglot':
        TRAIN_BUF = len(x_train) // BATCH_SIZE * BATCH_SIZE
        VAL_BUF = int(0.1 * len(x_test)) // BATCH_SIZE * BATCH_SIZE
        TEST_BUF = int(0.9 * len(x_test)) // BATCH_SIZE * BATCH_SIZE

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train[:TRAIN_BUF]).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices(x_test[:VAL_BUF]).shuffle(VAL_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test[VAL_BUF:VAL_BUF + TEST_BUF]).shuffle(TEST_BUF).batch(BATCH_SIZE)
    return train_dataset, val_dataset, test_dataset