import time
import bisect
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from keras.layers import *
from losses import prob_loss
from evaluate import mnist_test, mnist_sum_test


class DigitSamplerNet(tf.keras.Model):

    def __init__(self, N, batch_size=10, samples=2, grad_type='rloo', learning_rate=1e-3, val_samples=1000):
        super().__init__()
        self.dim = N
        self.batch_size = batch_size
        self.samples = samples
        self.grad_type = grad_type
        self.learning_rate = learning_rate
        self.val_samples = val_samples
        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(6, 5, activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Conv2D(16, 5, activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dense(10))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.logger = Logger()

        if self.grad_type == 'icr' or self.grad_type == 'advanced_icr' or self.grad_type == 'advanced_icr2':
            self.indecater_multiplier()

    def indecater_multiplier(self):
        mult = np.zeros([self.dim, 10, self.samples, self.batch_size, self.dim], dtype=np.int64)
        replacement = np.zeros([self.dim, 10, self.samples, self.batch_size, self.dim], dtype=np.int64)
        for i in range(self.dim):
            for j in range(10):
                mult[i, j, :, :, i] = 1
                replacement[i, j, :, :, i] = j
        self.icr_mult = mult
        self.icr_replacement = replacement 

    @tf.function
    def call(self, x):
        return self.model(x)
    
    @tf.function
    def joint_logits(self, x):
        x = tf.concat([x[:, i, :, :] for i in range(self.dim)], axis=0)
        x = self(x)
        return tf.stack([x[i * self.batch_size:(i + 1) * self.batch_size, :] for i in range(self.dim)], axis=1)
        
    @tf.function
    def rloo_grads(self, x, y, loss_fn):
        with tf.GradientTape() as tape:
            logits = self.joint_logits(x)
            d = tfp.distributions.Categorical(logits=logits)
            samples = d.sample(self.samples)
            f_sample = loss_fn(samples, y)
            log_p_sample = tf.reduce_sum(d.log_prob(samples), axis=-1)
            f_mean = tf.reduce_mean(f_sample, axis=0)

            rloo = tf.reduce_sum(log_p_sample * tf.stop_gradient(f_sample - f_mean), axis=0) / (self.samples - 1)
            rloo_prob = tf.stop_gradient(f_mean - rloo) + rloo
            loss = -tf.math.log(rloo_prob + 1e-8)
            loss = tf.reduce_mean(loss, axis=0)
            f_mean = tf.reduce_mean(f_mean, axis=0)

            return f_mean, tape.gradient(loss, self.trainable_variables)
    
    @tf.function
    def indecater_grads(self, x, y, loss_fn):
        with tf.GradientTape() as tape:
            logits = self.joint_logits(x)
            d = tfp.distributions.Categorical(logits=logits)
            samples = d.sample(self.samples)
            f_mean_b = tf.reduce_mean(loss_fn(samples, y), axis=0)

            outer_samples = tf.stack([samples] * 10, axis=0)
            outer_samples = tf.stack([outer_samples] * self.dim, axis=0)
            outer_samples = outer_samples * (1 - self.icr_mult) + self.icr_replacement
            outer_loss = loss_fn(outer_samples, y) # [dim, 10, samples, batch_size]
            variable_loss = tf.reduce_mean(outer_loss, axis=2) # [dim, 10, batch_size]
            variable_loss = tf.transpose(variable_loss, [2, 0, 1])
            indecater_expression = tf.stop_gradient(variable_loss) * tf.math.softmax(logits, axis=-1) # [batch_size, dim, 10]
            indecater_expression = tf.reduce_sum(indecater_expression, axis=-1) # [batch_size, dim]
            indecater_expression = tf.reduce_sum(indecater_expression, axis=-1) # [batch_size]

            icr_prob = tf.stop_gradient(f_mean_b - indecater_expression) + indecater_expression
            loss = -tf.math.log(icr_prob + 1e-8)
            loss = tf.reduce_mean(loss)
            f_mean = tf.reduce_mean(f_mean_b, axis=0)
            indecater_grad = tape.gradient(-tf.math.log(indecater_expression + 1e-8), self.trainable_variables)
        return f_mean, indecater_grad

    @tf.function
    def advanced_indecater_grads(self, x, y, loss_fn):
        with tf.GradientTape() as tape:
            logits = self.joint_logits(x)
            d = tfp.distributions.Categorical(logits=logits)
            samples = d.sample(self.samples * self.dim)
            f_mean_b = tf.reduce_mean(loss_fn(samples, y), axis=0)

            samples = tf.reshape(samples, [self.dim, self.samples, self.batch_size, self.dim])
            outer_samples = tf.stack([samples] * 10, axis=1)
            outer_samples = outer_samples * (1 - self.icr_mult) + self.icr_replacement
            outer_loss = loss_fn(outer_samples, y) # [dim, 10, samples, batch_size]
            variable_loss = tf.reduce_mean(outer_loss, axis=2)
            variable_loss = tf.transpose(variable_loss, [2, 0, 1])
            indecater_expression = tf.stop_gradient(variable_loss) * tf.math.softmax(logits, axis=-1)
            indecater_expression = tf.reduce_sum(indecater_expression, axis=-1)
            indecater_expression = tf.reduce_sum(indecater_expression, axis=-1)

            icr_prob = tf.stop_gradient(f_mean_b - indecater_expression) + indecater_expression
            loss = -tf.math.log(icr_prob + 1e-8)
            loss = tf.reduce_mean(loss)
            f_mean = tf.reduce_mean(f_mean_b, axis=0)
            return f_mean, tape.gradient(-tf.math.log(indecater_expression + 1e-8), self.trainable_variables)
        
    @tf.function
    def advanced_indecater_grads2(self, x, y, loss_fn):
        with tf.GradientTape() as tape:
            logits = self.joint_logits(x)
            d = tfp.distributions.Categorical(logits=logits)
            samples = d.sample(self.samples * self.dim * 10)
            f_mean_b = tf.reduce_mean(loss_fn(samples, y), axis=0)

            outer_samples = tf.reshape(samples, [self.dim, 10, self.samples, self.batch_size, self.dim])
            outer_samples = outer_samples * (1 - self.icr_mult) + self.icr_replacement
            outer_loss = loss_fn(outer_samples, y) # [dim, 10, samples, batch_size]
            variable_loss = tf.reduce_mean(outer_loss, axis=2)
            variable_loss = tf.transpose(variable_loss, [2, 0, 1])
            indecater_expression = tf.stop_gradient(variable_loss) * tf.math.softmax(logits, axis=-1)
            indecater_expression = tf.reduce_sum(indecater_expression, axis=-1)
            indecater_expression = tf.reduce_sum(indecater_expression, axis=-1)

            icr_prob = tf.stop_gradient(f_mean_b - indecater_expression) + indecater_expression
            loss = -tf.math.log(icr_prob + 1e-8)
            loss = tf.reduce_mean(loss)
            f_mean = tf.reduce_mean(f_mean_b, axis=0)
            return f_mean, tape.gradient(-tf.math.log(indecater_expression + 1e-8), self.trainable_variables)

    def grads(self, x, y, loss_fn):
        if self.grad_type == 'rloo':
            return self.rloo_grads(x, y, loss_fn)
        elif self.grad_type == 'reinforce':
            return self.reinforce_grads(x, y, loss_fn)
        elif self.grad_type == 'icr':
            return self.indecater_grads(x, y, loss_fn)
        elif self.grad_type == 'advanced_icr':
            return self.advanced_indecater_grads(x, y, loss_fn)
        elif self.grad_type == 'advanced_icr2':
            return self.advanced_indecater_grads2(x, y, loss_fn)

    def train(self, data, epochs, val_data=None, log_its=100, loss_fn=prob_loss):
        counter = 1
        acc_loss = 0
        var_grads = []
        prev_time = time.time()
        acc_time = 0
        for epoch in range(epochs):
            for x, y in data:
                loss, grad = self.grads(x, y, loss_fn)
                acc_loss += loss
                acc_time += time.time() - prev_time
                var_grads.append(tf.reduce_mean([tf.math.reduce_variance(g) for g in grad]))
                prev_time = time.time()
                self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
                if counter % log_its == 0:
                    acc_time += time.time() - prev_time
                    val_counter = 0
                    val_loss = 0
                    for val_x, val_y in val_data:
                        logits = self.joint_logits(val_x)
                        d = tfp.distributions.Categorical(logits=logits)
                        samples = d.sample(self.samples)
                        add_loss = tf.reduce_mean(loss_fn(samples, val_y))
                        val_loss += add_loss
                        val_counter += 1
                    mnist_acc = mnist_test(self)
                    mnist_sum_acc = mnist_sum_test(val_data, self)
                    print(
                        f"Epoch {epoch} iterations {counter}: {acc_loss / log_its}",
                        f"Validation loss: {val_loss / val_counter}",
                        f"MNIST accuracy: {mnist_acc}",
                        f"MNIST sum accuracy: {mnist_sum_acc}",
                        f"Time (s): {acc_time}",
                        f"Gradient variance: {np.mean(var_grads)}",
                        )
                    self.logger.log("training_loss", counter, acc_loss / log_its)
                    self.logger.log("validation_loss", counter, val_loss / val_counter)
                    self.logger.log("mnist_accuracy", counter, mnist_acc)
                    self.logger.log("mnist_sum_accuracy", counter, mnist_sum_acc)
                    self.logger.log("time", counter, acc_time)
                    self.logger.log("gradient_variance", counter, np.mean(var_grads))
                    acc_loss = 0
                    acc_time = 0
                    var_grads = []
                    prev_time = time.time()
                counter += 1


class Logger(object):

    def __init__(self):
        super(Logger, self).__init__()
        self.log_dict = dict()
        self.indices = list()

    def log(self, name, index, value):
        if name not in self.log_dict:
            self.log_dict[name] = dict()
        i = bisect.bisect_left(self.indices, index)
        if i >= len(self.indices) or self.indices[i] != index:
            self.indices.insert(i, index)
        self.log_dict[name][index] = value