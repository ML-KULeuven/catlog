## Discrete Variational Auto-Encoder
The network and all gradient estimators are implemented in the file [dave.py](https://github.com/LennertDeSmet/CatLog/blob/master/DVAE/dave.py). 
To run the experiment, simply pick the desired hyperparameters in the script [exp.py](https://github.com/LennertDeSmet/CatLog/blob/master/DVAE/exp.py) and execute it.

### Parameters
We elaborate on the possible choices of parameters in [exp.py](https://github.com/LennertDeSmet/CatLog/blob/master/DVAE/exp.py).  
As gradient estimator, one can pick IndeCater ('icr'), RLOO ('rloo') or the Gumbel-Softmax ('gs'). In our experiments the base number of samples for IndeCateR, GS-S and RLOO-S was two.
For the version with equal function evaluations to IndeCateR with 2 samples, being RLOO-F and GS-F, one can change the number of samples to 800 (2 * D * K).

Other parameters of note that can be chosen are the desired dataset, being MNIST ('mnist'), F-MNIST ('fmnist') and Omniglot ('omniglot'), or the learning rate. The other parameters are self-explanatory.
