import os
import pickle
import wandb
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import multiprocessing

from dave import Dave
from losses import elbo

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# hyperparameters
LR = 1e-4
DIM = 200
GRAD = 'icr'
TEMP = 1.
ANNEAL_RATE = 0.01
SAMPLES = 1
VAL_SAMPLES = 1000
BATCH_SIZE = 100
EPOCHS = 1000
DATA = 'fmnist'
CATS = 2

def train_dave(seed):
    # wandb
    wandb.init(
    # set the wandb project where this run will be logged
    project="IndeCateR",
    
    # track hyperparameters and run metadata
    config={
        "samples": SAMPLES,
        "experiment": "Gradient Estimation",
        "dataset": DATA,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "grad": GRAD,
        "run": seed,
        "cats": CATS,
        "dim": DIM
        }
    )

    # model
    dave = Dave(latent_dim=DIM, temp=TEMP, anneal_rate=ANNEAL_RATE, samples=SAMPLES, val_samples=VAL_SAMPLES, lr=LR, grad=GRAD, batch_size=BATCH_SIZE, cats=CATS)

    # data
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

    # train
    dave.train(train_dataset, EPOCHS, val_data=test_dataset, log_its=100)

    # test
    total_loss = 0
    test_counter = 0
    for x_test in test_dataset:
        z, x_hat = dave(x_test, VAL_SAMPLES)
        loss = tf.reduce_mean(elbo(x_test, x_hat, z), axis=0)
        loss = tf.reduce_mean(loss)
        total_loss += loss
        test_counter += 1
    print(f"Test elbo: {total_loss / test_counter}")
    dave.logger.log("test_elbo", -1, total_loss / test_counter)

    pickle.dump(dave.logger, open(f"DVAE/results_{DATA}/{GRAD}_dave_s{SAMPLES}_{DIM}_{EPOCHS}epochs_lr{LR}_{seed}_barabas.p", "wb"))


for i in range(1):
    p = multiprocessing.Process(target=train_dave, args=(i,))
    p.start()
    p.join()