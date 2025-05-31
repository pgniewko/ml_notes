# VQ-VAE: Vector Quantized Variational Autoencoder

This repository contains a minimal, clear implementation of the **VQ-VAE** (Vector Quantized Variational Autoencoder) model, alongside thorough explanations of the key architectural ideas, training mechanisms, loss functions, and practical considerations. The code is written in PyTorch and can be used as a foundation for further research or as an educational resource.

---

## Table of Contents

- [Overview](#overview)
- [How VQ-VAE Works](#how-vq-vae-works)
- [Losses in VQ-VAE](#losses-in-vq-vae)
  - [Why Commitment and Codebook Losses?](#why-commitment-and-codebook-losses)
- [Straight-Through Estimator (STE)](#straight-through-estimator-ste)
- [Posterior Collapse: Why VQ-VAE Avoids It](#posterior-collapse-why-vq-vae-avoids-it)
- [How to Sample New Examples](#how-to-sample-new-examples)
  - [USA Example: Sampling from the Codebook](#usa-example-sampling-from-the-codebook)
- [Minimal VQ-VAE Implementation](#minimal-vq-vae-implementation)
- [References](#references)

---

## Overview

**VQ-VAE** introduces discrete latent variables to the autoencoder family of models, enabling efficient and interpretable representations of high-dimensional data such as images, audio, and video.  
Unlike traditional VAEs (which use continuous latent spaces), VQ-VAE uses a *codebook* of learnable embeddings. Each input is encoded and then quantized to the nearest codebook vector, forming a sequence of discrete codes that the decoder uses to reconstruct the input.

Key highlights:

- **Discrete, learnable latent space** (the codebook)
- **No KL divergence** as in standard VAEs, thus no posterior collapse
- **Enables learning a prior over discrete codes** for advanced generative modeling (e.g., DALL·E, Jukebox, VQGAN)

---

## How VQ-VAE Works

The process can be summarized in three main steps:

1. **Encoder:**  
   Maps the input \( x \) to a continuous latent vector \( z_e(x) \).

2. **Vector Quantization:**  
   The encoder output \( z_e(x) \) is quantized to the closest codebook entry \( e_k \):

   \[
   k^* = \arg\min_k \| z_e(x) - e_k \|_2
   \]
   \[
   z_q(x) = e_{k^*}
   \]

   Here, the codebook \( \{e_1, e_2, ..., e_K\} \) contains \( K \) learnable vectors of dimension \( D \).

3. **Decoder:**  
   The decoder takes the quantized codebook vector \( z_q(x) \) and attempts to reconstruct the original input \( \hat{x} \).

---

## Losses in VQ-VAE

The VQ-VAE loss function consists of:

### 1. **Reconstruction Loss**

Measures how well the output matches the input.

\[
\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|_2^2
\]

### 2. **Commitment Loss**

Encourages the encoder outputs to be close to their assigned codebook vector, preventing the encoder from drifting arbitrarily far from the discrete representation.

\[
\mathcal{L}_{\text{commit}} = \beta \| z_e(x) - \text{sg}[e_{k^*}] \|_2^2
\]

- Here, \(\beta\) is a hyperparameter (e.g., 0.25).
- \(\text{sg}[\cdot]\) is the **stop-gradient** operation, ensuring this loss only updates the encoder.

### 3. **Codebook Loss**

Encourages each codebook entry to be close to the encoder outputs that use it, ensuring the codebook remains relevant and representative.

\[
\mathcal{L}_{\text{codebook}} = \| \text{sg}[z_e(x)] - e_{k^*} \|_2^2
\]

- Only the codebook entries receive gradients from this term.

### **Total Loss**

\[
\mathcal{L}_{\text{VQ-VAE}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{codebook}} + \mathcal{L}_{\text{commit}}
\]

---

### Why Commitment and Codebook Losses?

- **Commitment loss** keeps the encoder outputs close to the codebook, encouraging stable and efficient usage of discrete representations.  
- **Codebook loss** is essential because the quantization operation is non-differentiable. The encoder receives gradients via the straight-through estimator, but the codebook vectors themselves need explicit loss to move towards the encoder outputs.

---

## Straight-Through Estimator (STE)

The quantization step uses `argmin`, which is non-differentiable. Without special treatment, gradients cannot flow from the decoder through to the encoder. The **STE** is a practical trick that solves this:

- **Forward pass:** Replace encoder output with the nearest codebook entry.
- **Backward pass:** Copy the gradient from the decoder input back to the encoder output, as if quantization was an identity function.

**STE in code:**
```python
quantized = z_e + (quantized - z_e).detach()

