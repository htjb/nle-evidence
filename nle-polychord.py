import pypolychord
import numpy as np
import torch
from pypolychord.settings import  PolyChordSettings
from pypolychord.priors import UniformPrior
import matplotlib.pyplot as plt
import pickle
import mpi4py

density_estimator = pickle.load(open('density_estimator.pkl', 'rb'))
true_data = np.loadtxt('true_data.txt').reshape(1, 1).astype(np.float32)

def prior(cube):
    theta = np.zeros(len(cube))
    theta[0] = UniformPrior(1, 10)(cube[0])
    return theta

def likelihood(theta):
    return density_estimator.log_prob(torch.tensor(true_data),
                    torch.tensor([theta.astype(np.float32)])).detach().numpy()[0].astype(np.float64), []

nDims = 1

for i in range(2):
    settings = PolyChordSettings(nDims, 0) #settings is an object
    settings.read_resume = False
    settings.base_dir =  'testing-nle_' + str(i) + '/'
    settings.nlive = 100

    output = pypolychord.run_polychord(likelihood, nDims, 0, settings, prior)
    paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
    output.make_paramnames_files(paramnames)
