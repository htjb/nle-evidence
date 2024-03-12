import pypolychord
import numpy as np
import torch
from pypolychord.settings import  PolyChordSettings
from pypolychord.priors import UniformPrior
from anesthetic import read_chains
import matplotlib.pyplot as plt
import pickle
import mpi4py
from sbi.inference import SNLE
from sbi.utils.get_nn_models import likelihood_nn
from sbi import utils

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

np.random.seed(0)
torch.manual_seed(0)

def simulation(theta):
    return theta**(-1.5) + torch.normal(0, 0.05, size=theta.shape)

torch_prior = utils.BoxUniform(low=torch.tensor([1]),
                            high=torch.tensor([10]))

prior_sample = torch_prior.sample((1000,))

data = simulation(prior_sample)
data_mean = data.mean()
data_std = data.std()
norm_data = (data - data_mean) / data_std

for i in range(10):

    density_estimator_build_fun = likelihood_nn(
        model="maf", hidden_features=50,  
        num_transforms=2, z_score_x=None, z_score_theta='independent',
        use_batch_norm=True,
        )

    inference = SNLE(prior=torch_prior, density_estimator=density_estimator_build_fun)
    inference = inference.append_simulations(prior_sample, norm_data)
    density_estimator = inference.train()

    with open('density_estimator.pkl', 'wb') as f:
        pickle.dump(density_estimator, f)
    settings = PolyChordSettings(nDims, 0) #settings is an object
    settings.read_resume = False
    settings.base_dir =  'testing-nle_stability_' + str(i) + '/'
    settings.nlive = 100

    output = pypolychord.run_polychord(likelihood, nDims, 0, settings, prior)
    paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
    output.make_paramnames_files(paramnames)
    
    chains = read_chains('testing-nle_stability_' + str(i) + '/test')
    plt.hist(chains.logZ(5000), bins=20, alpha=0.5, label='NLE ' + str(i))

plt.xlabel('log(Z)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('nle-stability.png', dpi=300, bbox_inches='tight')
plt.show()
