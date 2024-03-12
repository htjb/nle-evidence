## Testing Evidences from NLEs

I wrote a simple example with one parameter and one data point to demonstrate that we can recover a good estimate of the Bayesian 
Evidence by sampling over a neural likelihood estimator (NLE). My data is modelled according to

$D = \theta^{-1.5} + x$

where $x \sim \mathcal{N}(0, 0.05)$ and my true data is generated with $\theta=2.5$.

![plot of data](https://github.com/htjb/nle-evidence/blob/main/data.png?raw=true)


Need to run in the following order:

- train-nle.ipynb which gives you the true data, the trained NLE and the scipy quad estimate of the evidence
    - note I am not doing any data or paraemter normalisation here so I don't need to worry about Jacobians
    - this is fine since the model is simple and the data and parameter are of order 1
- analytic-polychord.py which samples an analytic likelihood (from scipy.stats) twice
- nle-polychord.py which samples the NLE as a likelihood function twice
- compare_evidence.py which plots the evidences from the four nested sampling runs and the 'truth' from scipy.quad