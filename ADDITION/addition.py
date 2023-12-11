import os
import pickle
import multiprocessing
import tensorflow as tf

from network import DigitSamplerNet
from losses import prob_loss
from evaluate import mnist_sum_test
from data_generation import create_sum_loader


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

GRAD = 'advanced_icr'
BATCH_SIZE = 10
EPOCHS = 200
N = 32
# SAMPLES = 10 * N * 10
# SAMPLES = 10 * N
SAMPLES = 10
RUNS = 5

train_dataset, _, test_dataset = create_sum_loader(N, BATCH_SIZE=BATCH_SIZE)

for seed in range(RUNS):
    digit_sampler = DigitSamplerNet(N, samples=SAMPLES, grad_type=GRAD, batch_size=BATCH_SIZE)
    digit_sampler.train(train_dataset, EPOCHS, test_dataset, log_its=100, loss_fn=prob_loss)

    # test
    test_acc = mnist_sum_test(test_dataset, digit_sampler)
    print(f"Test accuracy: {round(test_acc.numpy() * 100, 3)} %")
    digit_sampler.logger.log('test_acc', -1, test_acc.numpy())

    pickle.dump(digit_sampler.logger, open(f"ADDITION/results_{GRAD}/samples{SAMPLES}_digits{N}_epochs{EPOCHS}_{seed}.p", "wb"))
