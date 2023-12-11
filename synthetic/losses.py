import tensorflow as tf
import numpy as np


def abs_loss(x, target):
    if x.dtype == tf.float32:
        return tf.reduce_sum(tf.abs(x - tf.cast(target, dtype=tf.float32)), axis=-1)
    return tf.cast(tf.reduce_sum(tf.abs(x - target), axis=-1), tf.float32)


def sq_loss(x, target, soft=False):
    if x.dtype == tf.float32:
        return -tf.reduce_mean((x - 0.499) ** 2, axis=-1)
    x = tf.cast(x, dtype=tf.float32)
    return -tf.reduce_mean((x - 0.499) ** 2, axis=-1)