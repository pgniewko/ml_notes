# Magnitude-Aware Dropout (Per-Sample Normalization)

## Overview

This repository introduces **Magnitude-Aware Dropout**, an extension of standard dropout where the probability 
of dropping an activation depends on its magnitude. Unlike standard dropout which is uniform, this variant ensures 
that the **normalization happens independently per sample** — i.e., each sample in a batch computes its own 
dropout probabilities across its feature dimensions.

## Method

1. **Per-Sample Normalization**  
   For a batch of activations \(x \in \mathbb{R}^{B \times D}\) or \(x \in \mathbb{R}^{B \times C \times H \times W}\), 
   we compute magnitudes \(m = |x|\). For each sample \(b\), we normalize across all non-batch dimensions:

   \[
   q_{b,i} = \frac{|x_{b,i}|}{\sum_j |x_{b,j}|}
   \]

   If all values are zero, \(q_{b,i} = 1/M\) where \(M\) is the number of elements in that sample.

2. **Keep Probabilities**  
   For each sample and element, we define keep probabilities:

   \[
   k_{b,i} = (1 - p) \Big[ (1 - \text{mix}) + \text{mix} \cdot M q_{b,i} \Big]
   \]

   where:
   - \(p\) is the global dropout rate,
   - \(M\) is the number of elements in the sample’s non-batch dimensions,
   - \(\text{mix}\) interpolates between standard dropout (\(0\)) and magnitude-based dropout (\(1\)).

   This ensures that for **each sample**:

   \[
   \frac{1}{M} \sum_i k_{b,i} = 1 - p.
   \]

3. **Sampling and Scaling**  
   We sample:

   \[
   z_{b,i} \sim Bernoulli(k_{b,i})
   \]

   and apply inverted dropout scaling:

   \[
   y_{b,i} = \frac{z_{b,i}}{k_{b,i}} x_{b,i}
   \]

   guaranteeing unbiasedness: \( \mathbb{E}[y_{b,i} | x] = x_{b,i} \).

## Novelty

- **Strict per-sample normalization**: dropout probabilities are normalized **within each sample**, never across the entire batch.
- **Interpolation with standard dropout**: controlled by `mix`, which allows smooth transition from uniform to magnitude-based dropout.
- **Expectation-preserving**: like standard dropout, the expected output remains equal to the input.
- **Standard dropout as a special case**: when `mix=0`, the method reduces exactly to uniform dropout.

## Examples

- **MLP input [B, D]**: normalization occurs separately for each of the B samples across D features.  
- **Conv input [B, C, H, W]**: normalization occurs separately for each sample across its (C, H, W) block.  
- **Zeros fallback**: if all magnitudes are zero in a sample, probabilities are uniform.

## Comparison to Prior Work

- Related to adaptive dropout and importance-based dropout, but this approach enforces the global constraint that each sample retains on average a \(1-p\) fraction of activations.
- Different from pruning: this is stochastic and unbiased, not a deterministic removal of low-magnitude units.
- Unlike variational dropout: no parameters are learned; probabilities are computed directly from activations.

## Applications

- Training deep networks with **signal-preserving dropout** that favors strong activations.  
- Can help reduce training noise by avoiding over-dropping informative neurons.  
- Usable in both fully connected and convolutional architectures.

---
