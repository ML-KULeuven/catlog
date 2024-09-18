import os
import pickle
import multiprocessing
import tensorflow as tf

from argparse import ArgumentParser
from network import DigitSamplerNet
from losses import prob_loss
from evaluate import mnist_sum_test
from data_generation import create_sum_loader


def run(method, batch_size, epochs, number_of_digits, samples, run):
    train_dataset, _, test_dataset = create_sum_loader(number_of_digits, BATCH_SIZE=batch_size)

    digit_sampler = DigitSamplerNet(number_of_digits, samples=samples, grad_type=method, batch_size=batch_size)
    digit_sampler.train(train_dataset, epochs, test_dataset, log_its=100, loss_fn=prob_loss)

    # test
    test_acc = mnist_sum_test(test_dataset, digit_sampler)
    print(f"Test accuracy: {round(test_acc.numpy() * 100, 3)} %")
    digit_sampler.logger.log('test_acc', -1, test_acc.numpy())

    pickle.dump(digit_sampler.logger, open(f"ADDITION/results_{method}/samples{samples}_digits{number_of_digits}_epochs{epochs}_{run}.p", "wb"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--method', type=str, default="advanced_icr")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--number_of_digits', type=int, default=16)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    for i in range(args.runs):
        run(args.method, args.batch_size, args.epochs, args.number_of_digits, args.samples, args.runs)