# Magnitude-Aware Dropout

## Overview

This repository introduces **Magnitude-Aware Dropout**, a novel extension of the standard dropout technique. 
While standard dropout assigns each activation an equal probability of being dropped, our approach assigns 
drop probabilities that depend on the **magnitude of activations**. Intuitively, units with larger magnitudes 
are more informative and should be dropped less often, while smaller activations should be dropped more often.

## Method

1. **Magnitude to Probability Mass**  
   For activations \(x_i\), we compute magnitudes \(m_i = |x_i|\) and normalize them into a probability 
   distribution \(q_i\).

   \[
   q_i = \frac{|x_i|}{\sum_j |x_j|}
   \]

   If all \(x_i = 0\), we default to a uniform distribution.

2. **Keep Probabilities**  
   The keep probability for each activation is defined as:

   \[
   k_i = (1 - p) \big[ (1 - \text{mix}) + \text{mix} \cdot M q_i \big]
   \]

   where:
   - \(p\) is the overall dropout rate,
   - \(M\) is the number of elements in the group,
   - \(\text{mix} \in [0,1]\) interpolates between standard dropout (uniform, \(\text{mix}=0\)) 
     and fully magnitude-based dropout (\(\text{mix}=1\)).

3. **Inverted Scaling**  
   After sampling \(z_i \sim Bernoulli(k_i)\), the output is:

   \[
   y_i = \frac{z_i}{k_i} x_i
   \]

   ensuring \(\mathbb{E}[y_i] = x_i\), making the transformation unbiased.

## Novelty

- **Mass-conserving normalization**: Magnitudes are turned into a probability mass that sums to 1, ensuring 
  principled probability assignments.
- **Interpolation with standard dropout**: A `mix` parameter allows smooth transition between uniform dropout 
  and magnitude-based dropout.
- **Expectation-preserving**: Just like standard dropout, the transformation preserves the expectation of 
  activations while redistributing dropout probabilities.
- **Special case equivalence**: Setting `mix=0` recovers standard dropout exactly, making the approach a 
  strict generalization.

## Comparison to Prior Work

- Related to **adaptive dropout** and **importance-based dropout**, but distinct because it guarantees the 
  same expected keep rate \((1-p)\) across activations.
- Unlike magnitude-based pruning, this method is **stochastic and unbiased**.
- Compared to variational dropout, our approach does not learn probabilities but derives them directly 
  from the input signal.

## Applications

- Training deep networks where preserving strong signals is critical.
- Potentially reducing gradient noise by preferentially keeping high-activation neurons.
- Can be used in both feedforward and recurrent architectures.

## Citation

If you use this method in your work, please cite this repository.

---
