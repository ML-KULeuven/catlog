import time
import wandb
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd

from keras.layers import *
from keras.activations import *
from utils import Logger
from losses import elbo, elbo_optim, cat_elbo


class Dave(tf.keras.Model):

    def __init__(self, latent_dim, lr, grad, anneal_rate, samples, val_samples, batch_size, temp, h1=384, h2=256, img_size=28, cats=2):
        super(Dave, self).__init__()
        self.dim = latent_dim
        self.lr = lr
        self.grad = grad
        self.anneal_rate = anneal_rate
        self.samples = samples
        self.val_samples = val_samples
        self.batch_size = batch_size
        self.temp = temp
        self.img_size = img_size
        self.cats = cats

        self.encoder = tf.keras.Sequential()
        self.encoder.add(Dense(h1, activation="relu"))
        self.encoder.add(Dense(h2, activation="relu"))
        self.encoder.add(Dense(self.dim, activation='linear'))

        self.decoder = tf.keras.Sequential()
        self.decoder.add(Dense(h2, activation="relu"))
        self.decoder.add(Dense(h1, activation="relu"))
        self.decoder.add(Dense(self.img_size ** 2, activation='linear'))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        if "icr" in self.grad:
            self.indecater_multiplier()
        self.logger = Logger()

    def indecater_multiplier(self):
        mult = np.zeros([self.cats, self.dim, self.samples, self.batch_size, self.dim], dtype=np.int64)
        replacement = np.zeros([self.cats, self.dim, self.samples, self.batch_size, self.dim], dtype=np.int64)
        for i in range(self.dim):
            mult[1, i, :, :, i] = 1
            mult[0, i, :, :, i] = 1
            replacement[1, i, :, :, i] = 1
            replacement[0, i, :, :, i] = 0
        self.indecater_mult = tf.constant(mult)
        self.indecater_replacement = tf.constant(replacement) 

    @tf.function
    def sample(self, z, sample_size, soft=False):
        if self.cats == 2:
            probs = tf.sigmoid(z)
            if soft:       
                soft_probs = tfd.RelaxedBernoulli(self.temp, logits=z).sample(sample_size)
                hard_probs = tf.cast(tf.round(soft_probs), dtype=tf.float32)
                diff_out = tf.stop_gradient(hard_probs - soft_probs) + soft_probs
                return diff_out
            return tfd.Bernoulli(probs=probs).sample(sample_size)
        else:
            probs = tf.nn.softmax(z)
            if soft:
                soft_probs = tfd.RelaxedOneHotCategorical(self.temp, probs=z).sample(sample_size)
                k = tf.shape(probs)[-1]
                hard_probs = tf.one_hot(tf.argmax(soft_probs, axis=-1), soft_probs.shape[-1])
                diff_out = tf.stop_gradient(hard_probs - soft_probs) + soft_probs
                diff_out = tf.squeeze(tf.matmul(diff_out, tf.expand_dims(tf.range(k, dtype=tf.float32), axis=-1)))
                return diff_out
            return tfd.Categorical(probs=probs).sample(sample_size)


    @tf.function
    def call(self, x, sample_size, soft=False):
        z = self.encoder(x)
        samples = self.sample(z, sample_size, soft)
        x_hat = self.decoder(samples)
        return z, x_hat

    @tf.function
    def indecater_grads(self, x):
        with tf.GradientTape() as tape:
            theta = self.encoder(x)
            samples = tf.cast(self.sample(theta, self.samples), dtype=tf.int64)
            x_hat = self.decoder(samples)
            b_loss = tf.reduce_mean(elbo(x, x_hat, theta), axis=0)
            loss = tf.reduce_mean(b_loss)

            outer_samples = tf.stack([samples] * self.dim, axis=0)
            outer_samples = tf.stack([outer_samples] * self.cats, axis=0)
            outer_samples = outer_samples * (1 - self.indecater_mult) + self.indecater_replacement
            outer_samples_1 = outer_samples[1]
            outer_samples_0 = outer_samples[0]

            outer_loss_1 = elbo(x, self.decoder(outer_samples_1), theta)
            outer_loss_0 = elbo(x, self.decoder(outer_samples_0), theta)
            variable_loss_1 = tf.transpose(tf.reduce_mean(outer_loss_1, axis=1))
            variable_loss_0 = tf.transpose(tf.reduce_mean(outer_loss_0, axis=1))

            indecater_expression = tf.reduce_sum(tf.stop_gradient(variable_loss_1) * tf.math.sigmoid(theta), axis=1)
            indecater_expression += tf.reduce_sum(tf.stop_gradient(variable_loss_0) * (1 - tf.math.sigmoid(theta)), axis=1)
            indecater_expression = tf.reduce_mean(indecater_expression)
            indecater_grad = tape.gradient(loss + indecater_expression, self.trainable_variables)
        return indecater_grad, loss
    
    # @tf.function
    # def indecater_grads(self, x, y, loss_fn):
    #     with tf.GradientTape() as tape:
    #         theta = self.encoder(x)
    #         samples = tf.cast(self.sample(theta, self.samples), dtype=tf.int64)
    #         x_hat = self.decoder(samples)
    #         b_loss = tf.reduce_mean(elbo(x, x_hat, theta), axis=0)
    #         loss = tf.reduce_mean(b_loss)

    #         outer_samples = tf.stack([samples] * self.dim, axis=0)
    #         outer_samples = tf.stack([outer_samples] * self.cats, axis=0)
    #         outer_samples = outer_samples * (1 - self.icr_mult) + self.icr_replacement
    #         outer_loss = elbo(x, self.decoder(outer_samples), theta) # [cats, dim, samples, batch_size]
    #         variable_loss = tf.reduce_mean(outer_loss, axis=2)
    #         variable_loss = tf.transpose(variable_loss, [2, 1, 0])
    #         indecater_expression = tf.stop_gradient(variable_loss) * tf.nn.softmax(theta, axis=-1) 
    #         indecater_expression = tf.reduce_sum(indecater_expression, axis=-1) 
    #         indecater_expression = tf.reduce_sum(indecater_expression, axis=-1)
    #     return indecater_expression, loss

    @tf.function
    def rloo_grads(self, x):
        with tf.GradientTape() as tape:
            theta = self.encoder(x)
            samples = tf.cast(self.sample(theta, self.samples), dtype=tf.int64)
            x_hat = self.decoder(samples)
            sample_probs = tf.where(samples == 1, tf.math.sigmoid(theta), 1 - tf.math.sigmoid(theta))
            sample_logps = tf.reduce_sum(tf.math.log(sample_probs), axis=-1)

            sample_loss = elbo(x, x_hat, theta)
            batch_loss = tf.reduce_mean(sample_loss, axis=0)
            loss = tf.reduce_mean(batch_loss)

            sample_rloo = tf.stop_gradient(sample_loss - batch_loss) * sample_logps
            rloo_expression = tf.reduce_sum(sample_rloo, axis=0) / (self.samples - 1)
            rloo_expression = tf.reduce_mean(rloo_expression)
            rloo_grad = tape.gradient(loss + rloo_expression, self.trainable_variables)
        return rloo_grad, loss

    @tf.function
    def gs_grads(self, x):
        with tf.GradientTape() as tape:
            theta = self.encoder(x)
            samples = self.sample(theta, self.samples, soft=True)
            x_hat = self.decoder(samples)
            loss = tf.reduce_mean(elbo(x, x_hat, theta), axis=0)
            loss = tf.reduce_mean(loss)
            gs_grad = tape.gradient(loss, self.trainable_variables)
            gs_grad = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in gs_grad]
        return gs_grad, loss
    
    @tf.function
    def advanced_indecater(self, x):
        with tf.GradientTape() as tape:
            theta = self.encoder(x)
            samples = tf.cast(self.sample(theta, self.samples * self.dim), dtype=tf.int64)
            x_hat = self.decoder(samples)
            b_loss = tf.reduce_mean(elbo(x, x_hat, theta), axis=0)
            loss = tf.reduce_mean(b_loss)

            outer_samples = tf.reshape(samples, [self.dim, self.samples, self.batch_size, self.dim])
            outer_samples = tf.expand_dims(outer_samples, axis=0)
            outer_samples = tf.repeat(outer_samples, self.cats, axis=0)
            outer_samples = outer_samples * (1 - self.indecater_mult) + self.indecater_replacement
            
            outer_samples = tf.reshape(outer_samples, [self.cats * self.dim * self.samples, self.batch_size, self.dim])
            x = tf.expand_dims(x, axis=0)
            outer_loss = elbo_optim(x, self.decoder(outer_samples), theta)
            outer_loss = tf.reshape(outer_loss, [self.cats, self.dim, self.samples, self.batch_size])

            variables_loss = tf.reduce_mean(outer_loss, axis=2)
            variables_loss = tf.transpose(variables_loss, [2, 0, 1])
            variable_loss_1 = tf.stop_gradient(variables_loss[:, 1, :])
            variable_loss_0 = tf.stop_gradient(variables_loss[:, 0, :])

            indecater_expression = tf.reduce_sum(variable_loss_1 * tf.math.sigmoid(theta), axis=1)
            indecater_expression += tf.reduce_sum(variable_loss_0 * (1 - tf.math.sigmoid(theta)), axis=1)
            indecater_expression = tf.reduce_mean(indecater_expression)
            indecater_grad = tape.gradient(loss + indecater_expression, self.trainable_variables)
        return indecater_grad, loss

    def grads(self, x):
        if self.grad == 'icr':
            grad, loss = self.indecater_grads(x)
        elif self.grad == 'gs':
            grad, loss = self.gs_grads(x)
        elif self.grad == 'rloo':
            grad, loss = self.rloo_grads(x)
        elif self.grad == 'advanced_icr':
            grad, loss = self.advanced_indecater(x)
        return grad, loss

    def train(self, data, epochs, val_data=None, log_its=100):
        counter = 1
        acc_loss = 0
        var_grads = [0]
        prev_time = time.time()
        acc_time = 0
        for epoch in range(epochs):
            self.temp *= tf.exp(-self.anneal_rate)
            self.temp = tf.clip_by_value(self.temp, 0.1, 1)
            for x in data:
                grad, loss = self.grads(x)
                acc_loss += loss
                acc_time += time.time() - prev_time
                var_grads.append(tf.reduce_mean([tf.math.reduce_std(g) ** 2 for g in grad]))
                prev_time = time.time()
                self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
                if counter % log_its == 0:
                    acc_time += time.time() - prev_time
                    val_counter = 0
                    val_loss = 0
                    for val_x in val_data:
                        val_theta, val_x_hat = self.call(val_x, self.val_samples)
                        add_loss = tf.reduce_mean(tf.reduce_mean(elbo(val_x, val_x_hat, val_theta), axis=0))
                        val_loss += add_loss
                        val_counter += 1
                    print(
                        f"Epoch {epoch} iterations {counter}: {acc_loss / log_its}",
                        f"Validation loss: {val_loss / val_counter}",
                        f"Time (s): {acc_time}",
                        f"Gradient variance: {np.mean(var_grads)}",
                        )
                    self.logger.log("training_loss", counter, acc_loss / log_its)
                    self.logger.log("validation_loss", counter, val_loss / val_counter)
                    self.logger.log("time", counter, acc_time)
                    self.logger.log("gradient_variance", counter, np.mean(var_grads))

                    wandb.log({"training_loss": acc_loss / log_its, "validation_loss": val_loss / val_counter, "time": acc_time, "gradient_variance": np.mean(var_grads)})
                    acc_loss = 0
                    acc_time = 0
                    var_grads = []
                    prev_time = time.time()
                counter += 1


class CategoricalDave(Dave):

    def __init__(self, dim, cats, lr, grad, anneal_rate, samples, val_samples, batch_size, temp, h1=384, h2=256, img_size=28):
        super(CategoricalDave, self).__init__(dim, lr, grad, anneal_rate, samples, val_samples, batch_size, temp, h1, h2, img_size, cats=cats)

        self.encoder = tf.keras.Sequential()
        self.encoder.add(Dense(h1, activation="relu"))
        self.encoder.add(Dense(h2, activation="relu"))
        self.encoder.add(Dense(self.dim * self.cats, activation='linear'))
        self.encoder.add(Reshape([self.dim, self.cats]))

        self.decoder = tf.keras.Sequential()
        self.decoder.add(Dense(h2, activation="relu"))
        self.decoder.add(Dense(h1, activation="relu"))
        self.decoder.add(Dense(self.img_size ** 2, activation='linear'))

    def indecater_multiplier(self):
        mult = np.zeros([self.cats, self.dim, self.samples, self.batch_size, self.dim], dtype=np.int64)
        replacement = np.zeros([self.cats, self.dim, self.samples, self.batch_size, self.dim], dtype=np.int64)
        for i in range(self.dim):
            for j in range(self.cats):
                mult[j, i, :, :, i] = 1
                replacement[j, i, :, :, i] = j
        self.indecater_mult = tf.constant(mult)
        self.indecater_replacement = tf.constant(replacement) 

    def call(self, x, sample_size, soft=False):
        theta = self.encoder(x)
        z = self.sample(theta, sample_size, soft)
        x_hat = self.decoder(z)
        return theta, x_hat
    
    def sample(self, theta, sample_size, soft=False):
        probs = tf.nn.softmax(theta, axis=-1)
        if soft:
            soft_probs = tfd.RelaxedOneHotCategorical(self.temp, probs=probs).sample(sample_size)
            k = tf.shape(probs)[-1]
            hard_probs = tf.one_hot(tf.argmax(soft_probs, axis=-1), soft_probs.shape[-1])
            diff_out = tf.stop_gradient(hard_probs - soft_probs) + soft_probs
            diff_out = tf.squeeze(tf.matmul(diff_out, tf.expand_dims(tf.range(k, dtype=tf.float32), axis=-1)))
            return diff_out
        return tfd.Categorical(probs=probs).sample(sample_size)
    
    def grads(self, x):
        if self.grad == 'icr':
            grad, loss = self.indecater_grads(x)
        elif self.grad == 'gs':
            grad, loss = self.gs_grads(x)
        elif self.grad == 'rloo':
            grad, loss = self.rloo_grads(x)
        elif self.grad == 'advanced_icr':
            grad, loss = self.advanced_indecater(x)
        return grad, loss
    
    @tf.function
    def indecater_grads(self, x):
        with tf.GradientTape() as tape:
            theta = self.encoder(x)
            samples = tf.cast(self.sample(theta, self.samples), dtype=tf.int64)
            x_hat = self.decoder(samples)
            b_loss = tf.reduce_mean(cat_elbo(x, x_hat, theta), axis=0)
            loss = tf.reduce_mean(b_loss)

            outer_samples = tf.stack([samples] * self.dim, axis=0)
            outer_samples = tf.stack([outer_samples] * self.cats, axis=0)
            outer_samples = outer_samples * (1 - self.indecater_mult) + self.indecater_replacement

            outer_loss = cat_elbo(x, self.decoder(outer_samples), theta) # [cats, dim, samples, batch_size]
            variable_loss = tf.reduce_mean(outer_loss, axis=2)
            variable_loss = tf.stop_gradient(tf.transpose(variable_loss, [2, 1, 0]))

            indecater_expression = tf.reduce_sum(variable_loss * tf.nn.softmax(theta, axis=-1), axis=-1)
            indecater_expression = tf.reduce_sum(indecater_expression, axis=-1)
            indecater_expression = tf.reduce_mean(indecater_expression)
            indecater_grad = tape.gradient(loss + indecater_expression, self.trainable_variables)
        return indecater_grad, loss
    
    @tf.function
    def gs_grads(self, x):
        with tf.GradientTape() as tape:
            theta = self.encoder(x)
            samples = self.sample(theta, self.samples, True)
            x_hat = self.decoder(samples)

            loss = tf.reduce_mean(cat_elbo(x, x_hat, theta), axis=0)
            loss = tf.reduce_mean(loss)

            gs_grad = tape.gradient(loss, self.trainable_variables)
            gs_grad = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in gs_grad]
        return gs_grad, loss
    
    @tf.function
    def rloo_grads(self, x):
        with tf.GradientTape() as tape:
            theta = self.encoder(x)
            samples = tf.cast(self.sample(theta, self.samples), dtype=tf.int64)
            x_hat = self.decoder(samples)

            sample_logps = tfd.Categorical(logits=theta).log_prob(samples)
            sample_logps = tf.reduce_sum(sample_logps, axis=-1)

            sample_loss = cat_elbo(x, x_hat, theta)
            batch_loss = tf.reduce_mean(sample_loss, axis=0)
            loss = tf.reduce_mean(batch_loss)


            sample_rloo = tf.stop_gradient(sample_loss - batch_loss) * sample_logps
            rloo_expression = tf.reduce_sum(sample_rloo, axis=0) / (self.samples - 1)
            rloo_expression = tf.reduce_mean(rloo_expression)
            rloo_grad = tape.gradient(loss + rloo_expression, self.trainable_variables)
        return rloo_grad, loss
    
    @tf.function
    def advanced_indecater(self, x):
        with tf.GradientTape() as tape:
            theta = self.encoder(x)
            samples = tf.cast(self.sample(theta, self.samples * self.dim), dtype=tf.int64)
            x_hat = self.decoder(samples)
            
            b_loss = tf.reduce_mean(cat_elbo(x, x_hat, theta), axis=0)
            loss = tf.reduce_mean(b_loss)

            outer_samples = tf.reshape(samples, [self.dim, self.samples, self.batch_size, self.dim])
            outer_samples = tf.expand_dims(outer_samples, axis=0)
            outer_samples = tf.repeat(outer_samples, self.cats, axis=0)
            outer_samples = outer_samples * (1 - self.indecater_mult) + self.indecater_replacement
            
            outer_loss = cat_elbo(x, self.decoder(outer_samples), theta) # [cats, dim, samples, batch_size]
            variable_loss = tf.reduce_mean(outer_loss, axis=2)
            variable_loss = tf.stop_gradient(tf.transpose(variable_loss, [2, 1, 0]))

            indecater_expression = tf.reduce_sum(variable_loss * tf.nn.softmax(theta, axis=-1), axis=-1)
            indecater_expression = tf.reduce_sum(indecater_expression, axis=-1)
            indecater_expression = tf.reduce_mean(indecater_expression)
            indecater_grad = tape.gradient(loss + indecater_expression, self.trainable_variables)
        return indecater_grad, loss


    def train(self, data, epochs, val_data=None, log_its=100):
        counter = 1
        acc_loss = 0
        var_grads = [0]
        prev_time = time.time()
        acc_time = 0
        for epoch in range(epochs):
            self.temp *= tf.exp(-self.anneal_rate)
            self.temp = tf.clip_by_value(self.temp, 0.1, 1)
            for x in data:
                grad, loss = self.grads(x)
                acc_loss += loss
                acc_time += time.time() - prev_time
                var_grads.append(tf.reduce_mean([tf.math.reduce_std(g) ** 2 for g in grad]))
                prev_time = time.time()
                self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
                if counter % log_its == 0:
                    acc_time += time.time() - prev_time
                    val_counter = 0
                    val_loss = 0
                    for val_x in val_data:
                        val_theta, val_x_hat = self.call(val_x, self.val_samples)
                        add_loss = tf.reduce_mean(tf.reduce_mean(cat_elbo(val_x, val_x_hat, val_theta), axis=0))
                        val_loss += add_loss
                        val_counter += 1
                    print(
                        f"Epoch {epoch} iterations {counter}: {acc_loss / log_its}",
                        f"Validation loss: {val_loss / val_counter}",
                        f"Time (s): {acc_time}",
                        f"Gradient variance: {np.mean(var_grads)}",
                        )
                    self.logger.log("training_loss", counter, acc_loss / log_its)
                    self.logger.log("validation_loss", counter, val_loss / val_counter)
                    self.logger.log("time", counter, acc_time)
                    self.logger.log("gradient_variance", counter, np.mean(var_grads))

                    wandb.log({"training_loss": acc_loss / log_its, "validation_loss": val_loss / val_counter, "time": acc_time, "gradient_variance": np.mean(var_grads)})
                    acc_loss = 0
                    acc_time = 0
                    var_grads = []
                    prev_time = time.time()
                counter += 1