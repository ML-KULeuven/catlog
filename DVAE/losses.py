import tensorflow as tf
import numpy as np


@tf.function
def elbo(x, x_hat, z):
    x_hat_l = len(x_hat.shape)
    x_l = len(x.shape)
    diff = x_hat_l - x_l
    while diff > 0:
        x = tf.stack([x] * x_hat.shape[diff - 1], axis=-1)
        x = tf.transpose(x, perm=[x_l] + [i for i in range(x_l)])
        x_l += 1
        diff -= 1
    cross_ent = x * tf.math.log_sigmoid(x_hat) + (1 - x) * tf.math.log_sigmoid(-x_hat)
    cross_ent = tf.math.reduce_sum(cross_ent, -1)

    q_z = tf.sigmoid(z)
    log_q_z = tf.math.log_sigmoid(z)
    prior_ent = q_z * (tf.math.log(0.5) - log_q_z) + (1 - q_z) * (tf.math.log(0.5) - tf.math.log_sigmoid(-z))
    prior_ent = tf.math.reduce_sum(prior_ent, -1)

    elbo = cross_ent + prior_ent # Should be [DIM, SAMPLES, BATCH]
    return -elbo

@tf.function
def cat_elbo(x, x_hat, z):
    cross_ent = x * tf.math.log_sigmoid(x_hat) + (1 - x) * tf.math.log_sigmoid(-x_hat)
    cross_ent = tf.math.reduce_sum(cross_ent, -1)

    q_z = tf.nn.softmax(z, axis=-1)
    log_q_z = tf.math.log_softmax(z, axis=-1)
    prior_ent = q_z * (tf.math.log(1 / 10) - log_q_z)
    prior_ent = tf.math.reduce_sum(prior_ent, -1)
    prior_ent = tf.math.reduce_sum(prior_ent, -1)

    elbo = cross_ent + prior_ent
    return -elbo

@tf.function
def elbo_optim(x, x_hat, z):
    cross_ent = x * tf.math.log_sigmoid(x_hat) + (1 - x) * tf.math.log_sigmoid(-x_hat)
    cross_ent = tf.math.reduce_sum(cross_ent, -1)

    q_z = tf.sigmoid(z)
    log_q_z = tf.math.log_sigmoid(z)
    prior_ent = q_z * (tf.math.log(0.5) - log_q_z) + (1 - q_z) * (tf.math.log(0.5) - tf.math.log_sigmoid(-z))
    prior_ent = tf.math.reduce_sum(prior_ent, -1)

    elbo = cross_ent + prior_ent # Should be [DIM, SAMPLES, BATCH]
    return -elbo