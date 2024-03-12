import pypolychord
import numpy as np
import torch
from pypolychord.settings import  PolyChordSettings
from pypolychord.priors import UniformPrior
from anesthetic import read_chains
import matplotlib.pyplot as plt
import pickle
import mpi4py

density_estimator = pickle.load(open('density_estimator.pkl', 'rb'))
true_data = np.loadtxt('true_data.txt').reshape(1, 1).astype(np.float32)
data_mean = np.loadtxt('data_mean.txt').reshape(1, 1).astype(np.float32)
data_std = np.loadtxt('data_std.txt').reshape(1, 1).astype(np.float32)
correction = np.loadtxt('logjacobian.txt')

norm_true_data = (true_data - data_mean) / data_std
norm_true_data = norm_true_data.astype(np.float32)

def prior(cube):
    theta = np.zeros(len(cube))
    theta[0] = UniformPrior(1, 10)(cube[0])
    return theta

def likelihood(theta):
    return (density_estimator.log_prob(torch.tensor(norm_true_data),
                    torch.tensor([theta.astype(np.float32)])).detach().numpy() \
                + correction).astype(np.float64)[0], []

print(likelihood(prior([0.5])))
nDims = 1

for i in range(2):
    settings = PolyChordSettings(nDims, 0) #settings is an object
    settings.read_resume = False
    settings.base_dir =  'testing-nle_' + str(i) + '/'
    settings.nlive = 100

    output = pypolychord.run_polychord(likelihood, nDims, 0, settings, prior)
    paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
    output.make_paramnames_files(paramnames)
    
    chains = read_chains('testing-nle_' + str(i) + '/test')
    print('NLE logZ:', chains.logZ(5000).mean())
