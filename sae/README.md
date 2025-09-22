# Sparse Autoencoder (SAE) – Relaxed Top-k Reimplementation

This repository explores **Sparse Autoencoders (SAEs)** as applied to protein embeddings. We follow ideas from recent work that use **Top-k SAEs** to disentangle sparse features, but with a few modifications to simplify training.

---

## Background: Sparse Autoencoders

Sparse Autoencoders (SAEs) are autoencoders designed to learn **disentangled, sparse features** from data. They constrain the hidden layer such that only a small fraction of neurons are active at a time. This sparsity improves interpretability and can help identify meaningful structure in the representations.

Two popular variants are:

- **L1 Sparse Autoencoder**  
  Adds an L1 penalty on latent activations. Sparsity is encouraged, but not guaranteed.

- **Top-k Sparse Autoencoder**  
  Forces exactly `k` activations per input to remain, zeroing out the rest. Sparsity is guaranteed, but gradients are trickier.

---

## Comparison: Top-k vs. L1 SAEs

| Aspect                  | **Top-k SAE**                                      | **L1 SAE**                                   |
|--------------------------|----------------------------------------------------|----------------------------------------------|
| **Sparsity Control**    | Exactly `k` active units per sample                | Number of active units varies; depends on λ   |
| **Interpretability**    | Strong: consistent number of features              | Moderate: sparsity pattern varies             |
| **Training Stability**  | Hard masking can kill neurons (dead features)      | Softer, easier to optimize                   |
| **Gradient Flow**       | Non-differentiable index selection (STE behavior)  | Fully differentiable                          |
| **Downstream Use**      | Fixed-size sparse codes (good for analysis)        | Adaptive sparsity (good for flexibility)      |

---

## Our Approach

We are inspired by the **Top-k SAE design** from the literature, but we make two key modifications:

1. **Soft Top-k Warmup**  
   - Instead of adding an **auxiliary reconstruction loss** (as in the paper) to mitigate dead neurons,  
     we start with a **soft/relaxed Top-k** using a **Gumbel-Softmax trick**.  
   - Over training, we **anneal the temperature** and **mix into hard Top-k**,  
     ending with exact sparse codes but with smoother gradients during warmup.

2. **Per-Sample LayerNorm**  
   - We normalize each input independently (across features, not across the batch).  
   - This makes inference **batch-independent** and stable without running statistics.

---

## Model Architecture

The model follows this structure:

```
x' = LayerNorm_per_sample(x)
z  = (x' - pre_bias) @ W_enc^T + latent_bias
h  = RelaxedTopK(z)                # soft → hard annealed Top-k
y' = decoder(h) + pre_bias
y  = unnormalize(y', mu, std)      # undo LayerNorm
```

Key points:
- **Pre-bias** (`pre_bias`) is subtracted before encoding and added back after decoding.  
- **Tied decoder** option: decoder weights can be tied to encoder transpose.  
- **Latent activation**: `RelaxedTopK` module starts soft and anneals to hard Top-k.

---

## Code Snippets

### Relaxed Top-k Activation

```python
class RelaxedTopK(nn.Module):
    def __init__(self, k, tau_start=1.0, tau_end=0.05, warmup_steps=2000):
        super().__init__()
        self.k = k
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.warmup_steps = warmup_steps
        self.register_buffer("step", torch.tensor(0))

    def forward(self, z):
        # Anneal temperature (tau) and hard-mix (alpha) over training steps
        t = min(self.step.item(), self.warmup_steps) / self.warmup_steps
        tau = self.tau_start * (1 - t) + self.tau_end * t
        alpha = t  # 0 → 1

        # Soft branch (Gumbel-Softmax over ReLU(z))
        z_pos = F.relu(z)
        g = -torch.log(-torch.log(torch.rand_like(z_pos) + 1e-9) + 1e-9)
        logits = (z_pos + g) / tau
        soft_probs = torch.softmax(logits, dim=-1)
        h_soft = soft_probs * z_pos.sum(dim=-1, keepdim=True)

        # Hard branch (standard Top-k)
        h_hard = hard_topk_relu(z, self.k)

        # Mix
        h = (1 - alpha) * h_soft + alpha * h_hard
        if self.training: self.step += 1
        return h
```

### Autoencoder Wrapper

```python
class Autoencoder(nn.Module):
    def __init__(self, n_latents, n_inputs, activation, tied=False, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation
        self.decoder = TiedTranspose(self.encoder) if tied else nn.Linear(n_latents, n_inputs, bias=False)

    def forward(self, x):
        x_norm, mu, std = LN(x) if self.normalize else (x, 0, 1)
        z = F.linear(x_norm - self.pre_bias, self.encoder.weight, self.latent_bias)
        h = self.activation(z)
        y = self.decoder(h) + self.pre_bias
        if self.normalize: y = y * std + mu
        return z, h, y
```

---

## Training Setup

- **Data**: We use **protein embeddings** (e.g. from ESM) stored in HDF5 files:
  ```
  Number of proteins: 20660
  First 5 IDs: ['A0A024R1R8', 'A0A024RBG1', 'A0A024RCN7', 'A0A075B6H5', 'A0A075B6H7']
  Embedding shape: (1024,)
  ```
- **Split**: 80% train, 10% validation, 10% test by protein ID.
- **Loss**: Reconstruction MSE.  
- **Metrics**: Track both MSE and MAE over time.  
- **Annealing**:  
  - τ (temperature) decays from `1.0 → 0.05`  
  - α (hard-mix) increases from `0 → 1`  
  over a warmup period (`~2000 steps`).

---

## Usage

The main training loop is implemented in a Jupyter notebook (`.ipynb`).  
It will:
- Load embeddings from HDF5  
- Train the model for `EPOCHS`  
- Log train/val/test MSE and MAE  
- Plot curves:  
  - `relaxed_topk_sae_mse.png`  
  - `relaxed_topk_sae_mae.png`  
- Save weights to `relaxed_topk_sae.pt`

See the notebook for full reproducibility.
