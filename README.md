## Testing Evidences from NLEs

In this branch I added data and parameter normalisation to see if I could correct for it properly.

I wrote a simple example with one parameter and one data point to demonstrate that we can recover a good estimate of the Bayesian 
Evidence by sampling over a neural likelihood estimator (NLE). My data is modelled according to

$D = \theta^{-1.5} + x$

where $x \sim \mathcal{N}(0, 0.05)$ and my true data is generated with $\theta=2.5$.

![plot of data](https://github.com/htjb/nle-evidence/blob/data_param_norm/data.png?raw=true)

I generate a set of simulated data from a Uniform prior on $\theta$ between 1 and 10 to train my NLE. The NLE is built with the
[sbi](https://github.com/sbi-dev/sbi) package. In this branch I have to worry about additional Jacobian factors from the data normalisation
when sampling over the NLE e.g.

$L(D|\tilde{\theta}) = L(\tilde{D}|\tilde{\theta}) |\frac{d \tilde{D}}{d D}|$

Because the likelihood is conditional on the parameters I dont think I need to correct for this normalisation. I
am using built in normalisation from the SBI package to normalise the parameters. It just uses standardisation.

My likelihood function is gaussian since the noise in the data is gaussian and I can define this with scipy.stats. 
I can analytically calcualte the evidence with scipy.integrate.quad, sample over scipy.stats.norm with
[Polychord](https://github.com/PolyChord/PolyChordLite) and sample over the NLE with Polychord too.

I use [anesthetic](https://anesthetic.readthedocs.io/en/latest/) to get $\log Z$ from the Nested Sampling runs and plot the distributions to compare with the
`quad` integration. I run the sampling over the NLE and `scipy.stats.norm` twice to check that they are
reproducible. You can see the results below.

My baselines are all computed in the real space.

![evidences](https://github.com/htjb/nle-evidence/blob/data_param_norm/evidence-comparison.png?raw=true)

### To run the code

Run in the following order:

- train-nle.ipynb which gives you the true data, the trained NLE and the scipy quad estimate of the evidence
- analytic-polychord.py which samples an analytic likelihood (from scipy.stats) twice
- nle-polychord.py which samples the NLE as a likelihood function twice
- compare_evidence.py which plots the evidences from the four nested sampling runs and the 'truth' from scipy.quad