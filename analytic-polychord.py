import pypolychord
import numpy as np
import torch
from pypolychord.settings import  PolyChordSettings
from pypolychord.priors import UniformPrior
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
import mpi4py

true_data = np.loadtxt('true_data.txt')

def prior(cube):
    theta = np.zeros(len(cube))
    theta[0] = UniformPrior(1, 10)(cube[0])
    return theta

def likelihood(theta):
    return norm.logpdf(theta**(-1.5), loc=true_data, scale=0.05).astype(np.float64)[0], []


nDims = 1

import pypolychord
from pypolychord.settings import  PolyChordSettings

for i in range(2):
    settings = PolyChordSettings(nDims, 0) #settings is an object
    settings.read_resume = False
    settings.base_dir =  'testing-stats-likelihood_' + str(i) + '/'
    settings.nlive = 100

    output = pypolychord.run_polychord(likelihood, nDims, 0, settings, prior)
    paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
    output.make_paramnames_files(paramnames)
