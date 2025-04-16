# Derivation of the posterioris for the Thompson Sampling

These notes are inspired by Patrick Walters's [paper](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01790) on the application of Thompson Sampling (TS) to virtual high-throughput screening (VHTS).

Our goal is to derive equations (4) and (5) from the paper using a Bayesian update for a univariate Gaussian model with a Gaussian prior.
In the paper, the derivation is provided for each reagent index $i$. 
Since these updates are independent, we drop the reagent index for simplicity.
