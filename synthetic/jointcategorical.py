import time
import bisect
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class JointCategorical(tf.keras.Model):

    def __init__(self, dims, cats, samples, temp=1.0, grad_type='rloo', anneal_rate=0.1):
        super(JointCategorical, self).__init__()
        self.dims = dims
        self.cats = cats
        self.samples = samples
        self.temp = temp
        self.grad_type = grad_type
        self.anneal_rate = anneal_rate
        self.logits = tf.Variable(tf.random.normal([dims, cats]))
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.logger = Logger()
        self.indecater_multiplier()

    def indecater_multiplier(self):
        mult = np.zeros([self.dims, self.cats, self.samples, self.dims], dtype=np.int32)
        replacement = np.zeros([self.dims, self.cats, self.samples, self.dims], dtype=np.int32)
        for i in range(self.dims):
            for j in range(self.cats):
                mult[i, j, :, i] = 1
                replacement[i, j, :, i] = j
        self.icr_mult = mult
        self.icr_replacement = replacement   

    @tf.function
    def call(self, samples=1, soft=False):
        if soft:
            soft_probs = tfp.distributions.RelaxedOneHotCategorical(self.temp, logits=self.logits).sample(samples)
            k = tf.shape(self.logits)[-1]
            hard_probs = tf.one_hot(tf.argmax(soft_probs, axis=-1), soft_probs.shape[-1])
            diff_out = tf.stop_gradient(hard_probs - soft_probs) + soft_probs
            diff_out = tf.squeeze(tf.matmul(diff_out, tf.expand_dims(tf.range(k, dtype=tf.float32), axis=-1)))
            return diff_out
        probs = tf.math.softmax(self.logits, axis=-1)
        distribution = tfp.distributions.Categorical(probs=probs)
        return distribution.sample(samples)

    def options(self):
        """ Returns all possible options for the joint distribution in the form of a tensor of shape [cats^dims, cats]
        """
        single_option = tf.expand_dims(tf.expand_dims(tf.range(self.cats), axis=0), axis=-1)
        total_options = tf.expand_dims(tf.expand_dims(tf.range(self.cats), axis=0), axis=0)
        for i in range(self.dims - 1):
            inter_option = tf.matmul(single_option, tf.ones([1, total_options.shape[-1]], dtype=tf.int32))
            total_options = tf.matmul(tf.ones([self.cats, 1], dtype=tf.int32), total_options)
            total_options = tf.reshape(tf.concat([total_options, inter_option], axis=0), [i + 2, 1, -1])
        return tf.transpose(tf.squeeze(total_options))

    def joint_probs(self, x, logits):
        probs = tf.math.softmax(logits, axis=-1)
        distribution = tfp.distributions.Categorical(probs=probs)
        return tf.reduce_prod(distribution.prob(x), axis=-1)

    def marginal_probs(self, x, logits):
        probs = tf.math.softmax(logits, axis=-1)
        distribution = tfp.distributions.Categorical(probs=probs)
        return distribution.prob(x)
    
    def grads(self, loss, target):
        if self.grad_type == "rloo":
            return self.rloo_grads(loss, target)
        elif self.grad_type == "reinforce":
            return self.reinforce_grads(loss, target)
        elif self.grad_type == "icr":
            return self.indecater_grads(loss, target)
        elif self.grad_type == 'gs':
            return self.gs_grads(loss, target)
    
    @tf.function
    def reinforce_grads(self, loss, target):
        with tf.GradientTape() as tape:
            samples = self(samples=self.samples)
            f_sample = loss(samples, target)
            log_p_sample = tf.reduce_sum(tf.math.log(self.marginal_probs(samples, self.logits) + 1e-20), axis=-1)
            f_mean = tf.reduce_mean(f_sample, axis=0)
            reinforce = tf.reduce_mean(log_p_sample * tf.stop_gradient(f_sample), axis=0)
            return f_mean, tape.gradient(reinforce, self.logits)
    
    @tf.function
    def rloo_grads(self, loss, target):
        with tf.GradientTape() as tape:
            samples = self(samples=self.samples)
            f_sample = loss(samples, target)
            log_p_sample = tf.reduce_sum(tf.math.log(self.marginal_probs(samples, self.logits) + 1e-20), axis=-1)
            f_mean = tf.reduce_mean(f_sample, axis=0)
            rloo = tf.reduce_sum(log_p_sample * tf.stop_gradient(f_sample - f_mean), axis=0) / (self.samples - 1)
            return f_mean, tape.gradient(rloo, self.logits)

    @tf.function
    def indecater_grads(self, loss, target):
        with tf.GradientTape() as tape:
            samples = self(samples=self.samples)
            f_mean = tf.reduce_mean(loss(samples, target))

            outer_samples = tf.stack([samples] * self.cats, axis=0)
            outer_samples = tf.stack([outer_samples] * self.dims, axis=0)
            outer_samples = outer_samples * (1 - self.icr_mult) + self.icr_replacement

            outer_loss = loss(outer_samples, target)
            variable_loss = tf.reduce_mean(outer_loss, axis=2)

            icr_expression = tf.stop_gradient(variable_loss) * tf.math.softmax(self.logits, axis=-1)
            icr_expression = tf.reduce_sum(icr_expression)
            icr_grad = tape.gradient(icr_expression, self.logits)
        return f_mean, icr_grad

    @tf.function
    def gs_grads(self, loss, target):
        with tf.GradientTape() as tape:
            gs_samples = self(samples=self.samples, soft=True)
            gs_loss = loss(gs_samples, target)
            f_mean = tf.reduce_mean(gs_loss, axis=0)
            gs_grad = tape.gradient(f_mean, self.logits)
            gs_grad = tf.where(tf.math.is_nan(gs_grad), tf.zeros_like(gs_grad), gs_grad)
            return f_mean, gs_grad

    def train(self, iterations, loss, target, log_its=100, learning_rate=1):
        acc_loss = 0
        acc_time = 0
        var_grads = [0]
        prev_time = time.time()
        for it in range(1, iterations + 1):
            if it % 20 == 0:
                self.temp *= tf.exp(-self.anneal_rate)
            f_mean, grads = self.grads(loss, target)
            acc_loss += f_mean
            acc_time += time.time() - prev_time
            var_grads.append(tf.reduce_mean([tf.math.reduce_variance(g) for g in grads]))
            prev_time = time.time()
            self.optimiser.apply_gradients(list(zip(grads, [self.logits])))
            if it % log_its == 0:
                acc_time += time.time() - prev_time
                print(
                    f"Iterations {it}: mean function {acc_loss / log_its}",
                    f"Time (s): {acc_time}",
                    f"Gradient variance: {np.mean(var_grads)}",
                    )
                self.logger.log("training_loss", it, acc_loss / log_its)
                self.logger.log("time", it, acc_time)
                self.logger.log("gradient_variance", it, np.mean(var_grads))
                acc_loss = 0
                acc_time = 0
                var_grads = []
                prev_time = time.time()


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