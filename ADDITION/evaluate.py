import tensorflow as tf


def mnist_test(model):
    """
    Test the model on the MNIST dataset.
    :param test_set: The MNIST test set.
    :param model: The model to test.
    :return: The accuracy of the model on the test set.
    """
    correct = 0
    total = 0
    test_x, test_y = tf.keras.datasets.mnist.load_data()[1]
    test_x, test_y = test_x.astype('float32') / 255., test_y.astype('float32')
    test_x = tf.expand_dims(test_x, axis=-1)
    test_set = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(100)
    for x, y in test_set:
        y_hat = model(x)
        correct += tf.math.count_nonzero(tf.math.equal(tf.math.argmax(y_hat, axis=-1), tf.math.argmax(y, axis=-1)))
        total += y.shape[0]
    return correct / total

def mnist_sum_test(test_set, model):
    correct = 0
    total = 0
    for x, y in test_set:
        y_hat = model.joint_logits(x)
        predicted_digits = tf.math.argmax(y_hat, axis=-1)
        predicted_sum = tf.cast(tf.math.reduce_sum(predicted_digits, axis=-1), tf.int32)
        correct += tf.math.count_nonzero(tf.math.equal(predicted_sum, y))
        total += y.shape[0]
    return correct / total