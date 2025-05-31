# VQ-VAE: Vector Quantized Variational Autoencoder

This repository demonstrates a minimal [implementation](./notebook/VQVAE-example.ipynb) of **VQ-VAE** (Vector Quantized Variational Autoencoder) in PyTorch, along with explanations for the key components of the model, including the straight-through estimator (STE), commitment loss, and codebook loss.

---

## Overview

**VQ-VAE** introduces discrete latent variables to the autoencoder framework. Instead of mapping inputs to a continuous latent space, the encoder outputs are quantized to the nearest entry in a learned codebook of embeddings. This enables the use of discrete representations for efficient compression and generation, and avoids issues present in standard VAEs.

---

## How VQ-VAE Works

### 1. Encoder

The encoder maps input data $\( x \)$ to a latent representation $\( z_e(x) \)$.

### 2. Vector Quantization

The encoder output $\( z_e(x) \)$ is quantized to the nearest codebook entry $\( e_k \)$:

$$
k^* = \arg\min_k \| z_e(x) - e_k \|_2
$$

$$
z_q(x) = e_{k^*}
$$

Where the codebook is a learnable matrix of $\( K \)$ entries, each of dimension $\( D \)$.

### 3. Decoder

The decoder reconstructs the original input from the quantized latent $\( z_q(x) \)$.

---

## Losses in VQ-VAE

VQ-VAE uses a combination of three loss terms:

### 1. Reconstruction Loss

This loss ensures that the reconstructed output is similar to the original input.

 $$
 \mathcal{L}_{\text{recon}} = \| x - \hat{x} \|_2^2
 $$

### 2. Commitment Loss

Encourages the encoder outputs to stay close to their assigned codebook entry, stabilizing training and ensuring efficient codebook usage.



  $$
  \mathcal{L}_{\text{commit}} = \beta \| z_e(x) - \text{sg}[e_k^*] \|_2^2
  $$

Where:
- $\beta$ is a hyperparameter.
- $\text{sg}[\cdot]$ is the stop-gradient operator, meaning gradients do not flow through this argument.
- $e_k^*$ is the nearest neighbor code.

### 3. Codebook Loss

Directly updates the codebook vectors to move them toward the encoder outputs that select them.

$$
\mathcal{L}_{\text{codebook}} = \| \text{sg}[z_e(x)] - e_k^* \|_2^2
$$

Where only the codebook entries are updated by this term.

### Total Loss

The total loss for VQ-VAE training is:

 $$
  L_{\text{VQ-VAE}} = L_{\text{recon}} + L_{\text{codebook}} + L_{\text{commit}}
 $$

---

## Why Commitment and Codebook Losses?

- **Commitment loss** is necessary because the straight-through estimator allows gradients to flow only to the encoder, not to the codebook. This term ensures the encoder outputs do not drift too far from the codebook vectors.
- **Codebook loss** is needed to move the embedding vectors toward the encoder outputs that select them, since gradients do not pass through the non-differentiable quantization operation.

---

## Straight-Through Estimator (STE)

The quantization step involves an `argmin` operation, which is non-differentiable. The **Straight-Through Estimator** is a trick used to allow gradients to flow from the decoder to the encoder as if quantization was the identity function (i.e. like quantization was never there).  
In code, this typically looks like:

```python
quantized = z_e + (quantized - z_e).detach()
```


This means that, in the backward pass, the gradient with respect to the quantized vector is copied directly to the encoder output.

---

## Reference
- [Neural Discrete Representation Learning (VQ-VAE original paper)](https://arxiv.org/abs/1711.00937)
