import tensorflow as tf


def prob_loss(x, target_sum):
    return tf.where(tf.reduce_sum(x, axis=-1) == target_sum, 1., 0.)

