# Sparse Autoencoder (SAE) 

## L1-SAE & Top‑K SAE for Protein Embeddings

This repo explores **Sparse Autoencoders (SAEs)** on per‑protein embeddings (`ProtT5`). It reimplements a small but practical SAE stack with:
- A **simple (ReLU) SAE** using an L1 penalty,
- A **hard Top‑K** sparse activation,

We keep the code compact and reproducible, while adopting several training and initialization tricks inspired by recent SAE implementations.

---

## What’s in this repo

- **`SparseAutoEncoder`**: minimal AE with optional **tied decoder weights** (decoder = encoderᵀ), and **non‑trainable per‑sample layer normalization**.
- **Activations**:
  - **ReLU** (baseline)
  - **`TopK(k)`**: keeps the top‑k activations per sample and zeroes out the rest
- **Training script / notebook cells** that:
  - Load HDF5 protein embeddings
  - Make an **80/10/10 train/val/test split**
  - Train **three models** and compare:
    1) **ReLU+L1** (no Top‑K) — loss = NMSE + $\lambda \cdot L_1$
    2) **TopK=64** (no L1) — loss = NMSE  
  - Log **loss** and **NMSE** on train/val/test
  - Plot a **comparison figure** across all models and splits

> **Note**: This code **does not use the auxiliary loss** sometimes employed in Top‑K SAEs to mitigate dead neurons.
---

## Background: Why Sparse AEs?

SAEs aim to learn **disentangled, sparse features**, where only a small subset of latent units are active for any given input. The resulting codes are more interpretable and can reveal meaningful structure in the representation space (e.g., functional motifs in protein embeddings).

Two mainstream variants:

- **L1 Sparse Autoencoder**: adds an L1 penalty on latent activations. Sparsity is encouraged but not guaranteed; the number of active units varies with $\lambda$ and data scale.
- **Top‑K Sparse Autoencoder**: enforces **exactly k** active units per sample (hard sparsity). Guarantees sparsity and yields fixed‑sized sparse codes, but the hard selection makes gradient flow trickier.

---

## Activations: ReLU vs Top‑K

### TL;DR comparison

| Aspect | **ReLU + L1** | **Top‑K** |
|---|---|---|
| **Sparsity** | Soft; controlled by λ; count varies | **Exactly `k`** actives per sample |
| **Gradients** | Smooth, standard | Non‑diff. indices (uses STE‑style masking) |
| **Stability** | Usually stable | Can create **dead neurons** if poorly initialized |
| **Interpretability** | Moderate | High: fixed sparsity aids analysis |
| **Hyperparams** | Choose λ | Choose k |

### When is the simple ReLU+L1 enough?
- If you want a **baseline** or prefer **fully differentiable** training with fewer moving parts.
- If exact `k`‑sparsity is not critical, and you’re comfortable tuning λ to get a desired sparsity regime.
- Often a strong, simple baseline — we include it for direct comparison.

  
### Why (and when) use Top‑K?
- Guarantees a **fixed number of active features** — convenient for downstream analysis or indexing specific “concept neurons”.
- Encourages competition among features; often yields **crisper, more interpretable** directions.
- Caveat: Naïve training may lead to **dead units** (never selected). Good initialization and occasional re‑normalization help.

---

## Architecture (high level)

```
x_norm, μ, σ = layer_norm_no_affine(x)        # per‑sample LN (no learned γ/β)
z_pre = (x_norm - pre_bias) @ W_encᵀ + b_lat  # encoder (no bias in W_enc)
h = activation(z_pre)                         # ReLU | TopK(k)
y_pre = decoder(h) + pre_bias                 # decoder (tied: W_dec = W_encᵀ)
y = y_pre * σ + μ                             # undo normalization
```

Implementation details adopted for stability & performance:
- **Kaiming/He init** for encoder, **tied decoder** (decoder = encoderᵀ).
- Optional **row‑norm** of decoder (or column‑norm of encoder if tied) at init.
- **Per‑sample LN** to reduce scale sensitivity.
- Training‑side ergonomics: **grad clipping**, **cosine LR + warmup**, **pin_memory / num_workers** in loaders, and deterministic seeding.

---

## Data & Splits

We read per‑protein embeddings from an HDF5 file (keys are UniProt IDs; values are vectors):

```python
import h5py, numpy as np
with h5py.File("./data/per-protein.h5", "r") as f:
    ids = list(f.keys())
    x = np.stack([f[k][()] for k in ids])  # [N, D]
```

We create an **80/10/10 split** (train/val/test) with a fixed seed for reproducibility.

---

## Losses & Metrics

We report two metrics per split:
- **Loss** (as optimized):  
  - ReLU+L1 model: `NMSE + λ·L1(latents)`  
  - TopK model: `NMSE` (no L1)
- **NMSE** (Normalized Mean Squared Error):

$$
\text{NMSE} = \mathbb{E}_b\left[ \frac{\|\hat{x}_b - x_b\|_2^2}{\|x_b\|_2^2} \right]
$$

- **The L1 term**:  

$$
\text{L1} = \mathbb{E}_b\left[ \frac{\|h_b\|_1}{\|x_b\|_2} \right]
$$

> **No auxiliary loss**: We deliberately **omit** the auxiliary reconstruction term sometimes used with Top‑K SAEs for reviving dead latents. With good init, training is stable in our setting. Feel free to re‑enable an aux loss if your domain benefits from it.

---

## Training the two variants

We train:
1. **ReLU+L1** (no Top‑K): `λ=1e-3` (tunable), `normalize=False`  
2. **TopK=64**: `normalize=True`, no `L1` 

For each model we collect **Loss** and **NMSE** on **train / val / test** every epoch and produce a **comparison plot** (2 rows × 3 columns; bars for each model).

---

## Quickstart

1) Put your HDF5 embeddings at `./data/per-protein.h5`  
2) Run the notebook or the provided script cell that:
   - builds dataloaders,
   - instantiates the three models,
   - trains for `EPOCHS` with Adam + cosine schedule,
   - logs/plots results.

> Hyperparameters: `n_latents`, `k`, `λ (L1)`, LR, batch size are all exposed in the code.

---

## Notes on implementation details we adopted

- **Encoder init**: Kaiming uniform (good with ReLU‑like activations).  
- **Tied decoder**: fewer parameters and an inductive bias toward symmetric encode/decode.  
- **Row‑norm at init**: stabilizes early optimization (common in SAE repos).  
- **Per‑sample LN** (`layer_norm_no_affine`) instead of `nn.LayerNorm`: we explicitly **don’t learn $\gamma$, $\beta$** to keep normalization purely geometric and avoid re‑centering by the network.
- **No auxiliary loss**: intentionally left out for simplicity; swap in if needed.

---

## Results (what to expect)

- **ReLU+L1** usually trains smoothly; sparsity depends on $\lambda$ and data scale.  
- **TopK=64** yields **consistent sparsity** and often **clearer features**, but may require careful init to avoid dead units.  

Exact numbers depend on your embeddings and training budget. See the final comparison figure in the notebook for your run.

---

## References

**Core inspiration**  
- *OpenAI*: **Scaling and evaluating sparse autoencoders** — code & ideas for Top‑K training and evaluation.  
  Repo: https://github.com/openai/sparse_autoencoder

**Domain application**  
- *PNAS*: **Sparse autoencoders uncover biologically interpretable features in protein language model representations** — demonstrates biologically meaningful features in protein embeddings.  
  Repo: https://github.com/onkarsg10/Rep_SAEs_PLMs

<!-- Additional repos that influenced practical details (init, normalization, Top‑K mechanics, training ergonomics):   -->
<!-- - https://github.com/DanielKerrigan/saefarer   -->
<!-- - https://github.com/EleutherAI/sae   -->
<!-- - https://github.com/bartbussmann/BatchTopK   -->
<!-- - https://github.com/tylercosgrove/sparse-autoencoder-mistral7b   -->
<!-- - https://github.com/jbloomAus/SAELens   -->
<!-- - https://github.com/neelnanda-io/1L-Sparse-Autoencoder -->

---

## License

See `LICENSE` for details.
