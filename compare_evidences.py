import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains

for i in range(2):
    nle_chains = read_chains('testing-nle_' + str(i) + '/test')
    stats_chains = read_chains('testing-stats-likelihood_' + str(i) + '/test')

    nle_logZ = nle_chains.logZ(5000)
    stats_logZ = stats_chains.logZ(5000)

    plt.hist(nle_logZ, bins=20, alpha=0.5, label='NLE ' + str(i))
    plt.hist(stats_logZ, bins=20, alpha=0.5, label='Stats ' + str(i))

true_logZ = np.loadtxt('raw-integration-evidence.txt')

plt.axvline(true_logZ, color='k', label='True')
plt.xlabel('log(Z)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('evidence-comparison.png')
plt.show()