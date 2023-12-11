## Neural-Symbolic MNIST Addition
This folder contains the experimental setup to learn to sum sequences of MNIST digits.  
This problem is challenging because of both its sparse supervision and its exponential sample space.

### Data generation
Data is generated in [data_generation.py]() by importing the standard MNIST dataset available from the TensorFlow package.  
All MNIST images are shuffled and sequences of the desired length are constructed afterwards by concatenating the images and summing up their labels.

### Experiment
The experiment uses a simple neural network provided in [network.py](https://github.com/LennertDeSmet/CatLog/blob/master/ADDITION/network.py) to identify each MNIST image in the given sequence. 
These predictions are then summed up and compared to the supervision.
Evaluation measures the accuracy of predicting the correct sum using the method from [evaluate.py](https://github.com/LennertDeSmet/CatLog/blob/master/ADDITION/evaluate.py). 
The total experiment can be run by executing [addition.py](https://github.com/LennertDeSmet/CatLog/blob/master/ADDITION/addition.py), that is, learning the networks based on various gradient estimators of choice.

#### Parameters
Similar to the DVAE experiment in [DVAE](https://github.com/LennertDeSmet/CatLog/tree/master/DVAE), there are a number of parameters that can be altered in the script [addition.py](https://github.com/LennertDeSmet/CatLog/blob/master/ADDITION/addition.py).
To change how to compute the gradient, simply change the variable `GRAD` to 'rloo', 'icr' or 'advanced_icr' for RLOO, IndeCateR or IndeCateR-D, respectively. Changing the variable `SAMPLES` changes the number of samples that each method takes.
To run results for RLOO-F, use `SAMPLES = 10 * N * 10`. For all other methods, use `SAMPLES = 10`. In this last case, IndeCateR-D will automatically draw 10 samples for each variable, hence drawing `10 * N` samples in total.
