import os
import pickle
import wandb
import numpy as np
import tensorflow as tf
import multiprocessing

from dave import CategoricalDave
from losses import cat_elbo
from data import load_data

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# hyperparameters
LR = 1e-4
DIM = 200
CATS = 10
GRAD = 'rloo'
TEMP = 1.
ANNEAL_RATE = 0.01
# SAMPLES = 1
SAMPLES = DIM
# SAMPLES = DIM * CATS
VAL_SAMPLES = 1000
BATCH_SIZE = 100
EPOCHS = 1000
DATA = 'mnist'

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
    dave = CategoricalDave(dim=DIM, cats=CATS, temp=TEMP, anneal_rate=ANNEAL_RATE, samples=SAMPLES, val_samples=VAL_SAMPLES, lr=LR, grad=GRAD, batch_size=BATCH_SIZE)

    # train
    train_dataset, val_dataset, test_dataset = load_data(DATA, BATCH_SIZE)
    dave.train(train_dataset, EPOCHS, val_data=test_dataset, log_its=100)

    # test
    total_loss = 0
    test_counter = 0
    for x_test in test_dataset:
        z, x_hat = dave(x_test, VAL_SAMPLES)
        loss = tf.reduce_mean(cat_elbo(x_test, x_hat, z), axis=0)
        loss = tf.reduce_mean(loss)
        total_loss += loss
        test_counter += 1
    print(f"Test elbo: {total_loss / test_counter}")
    dave.logger.log("test_elbo", -1, total_loss / test_counter)

    pickle.dump(dave.logger, open(f"DVAE/results_{DATA}/cat/{GRAD}_dave_s{SAMPLES}_{DIM}_{EPOCHS}epochs_lr{LR}_{seed}_{CATS}cats_barabas.p", "wb"))


for i in range(5):
    p = multiprocessing.Process(target=train_dave, args=(i,))
    p.start()
    p.join()