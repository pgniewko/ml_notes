# Sparse Autoencoder (SAE) – Top‑K & Gumbel‑Top‑K for Protein Embeddings

This repo explores **Sparse Autoencoders (SAEs)** on per‑protein embeddings (e.g., PLM/ESM‑style). It reimplements a small but practical SAE stack with:
- A **simple (ReLU) SAE** using an L1 penalty,
- A **hard Top‑K** sparse activation,
- A **Gumbel‑Top‑K** activation that anneals from soft/stochastic to hard Top‑K.

We keep the code compact and reproducible, while adopting several training and initialization tricks inspired by recent SAE implementations.

---

## What’s in this repo

- **`SparseAutoEncoder`**: minimal AE with optional **tied decoder weights** (decoder = encoderᵀ), and **non‑trainable per‑sample layer normalization**.
- **Activations**:
  - **ReLU** (baseline)
  - **`TopK(k)`**: keeps the top‑k activations per sample and zeroes out the rest
  - **`GumbelTopK(k)`**: samples Top‑K with Gumbel noise and **linearly anneals** temperature each forward pass, converging to hard Top‑K
- **Training script / notebook cells** that:
  - Load HDF5 protein embeddings
  - Make an **80/10/10 train/val/test split**
  - Train **three models** and compare:
    1) **ReLU+L1** (no Top‑K) — loss = NMSE + λ·L1  
    2) **TopK=64** (no L1) — loss = NMSE  
    3) **GumbelTopK=64** (no L1) — loss = NMSE
  - Log **loss** and **NMSE** on train/val/test
  - Plot a **comparison figure** across all models and splits

> **Note**: This code **does not use the auxiliary loss** sometimes employed in Top‑K SAEs to mitigate dead neurons. We found the annealed GumbelTopK and better initialization already provide stable training; feel free to add an aux loss if you want to match specific papers exactly.

---

## Background: Why Sparse AEs?

SAEs aim to learn **disentangled, sparse features**, where only a small subset of latent units are active for any given input. The resulting codes are more interpretable and can reveal meaningful structure in the representation space (e.g., functional motifs in protein embeddings).

Two mainstream variants:

- **L1 Sparse Autoencoder**: adds an L1 penalty on latent activations. Sparsity is encouraged but not guaranteed; the number of active units varies with λ and data scale.
- **Top‑K Sparse Autoencoder**: enforces **exactly k** active units per sample (hard sparsity). Guarantees sparsity and yields fixed‑sized sparse codes, but the hard selection makes gradient flow trickier.

---

## Activations: ReLU vs Top‑K vs Gumbel‑Top‑K

### TL;DR comparison

| Aspect | **ReLU + L1** | **Top‑K** | **Gumbel‑Top‑K** |
|---|---|---|---|
| **Sparsity** | Soft; controlled by λ; count varies | **Exactly `k`** actives per sample | Starts soft/stochastic → **exactly `k`** as τ→0 |
| **Gradients** | Smooth, standard | Non‑diff. indices (uses STE‑style masking) | Smooth early via Gumbel softmax; becomes hard |
| **Stability** | Usually stable | Can create **dead neurons** if poorly initialized | **Mitigates dead neurons** during warmup |
| **Interpretability** | Moderate | High: fixed sparsity aids analysis | High: same as Top‑K at convergence |
| **Hyperparams** | Choose λ | Choose k | Choose k + anneal schedule (τ_start, τ_end, steps) |

### Why (and when) use Top‑K?
- Guarantees a **fixed number of active features** — convenient for downstream analysis or indexing specific “concept neurons”.
- Encourages competition among features; often yields **crisper, more interpretable** directions.
- Caveat: Naïve training may lead to **dead units** (never selected). Good initialization and occasional re‑normalization help.

### Why (and when) use Gumbel‑Top‑K?
- Provides a **stochastic, differentiable relaxation** early on via Gumbel sampling + temperature **τ**.
- We **anneal τ linearly** each forward call; when τ ≈ τ_end (tiny), the selection is effectively **hard Top‑K**.
- This **reduces dead‑neuron risk** and improves early optimization while converging to the same sparsity pattern as hard Top‑K.

### When is the simple ReLU+L1 enough?
- If you want a **baseline** or prefer **fully differentiable** training with fewer moving parts.
- If exact `k`‑sparsity is not critical, and you’re comfortable tuning λ to get a desired sparsity regime.
- Often a strong, simple baseline — we include it for direct comparison.

---

## Architecture (high level)

```
x_norm, μ, σ = layer_norm_no_affine(x)        # per‑sample LN (no learned γ/β)
z_pre = (x_norm - pre_bias) @ W_encᵀ + b_lat  # encoder (no bias in W_enc)
h = activation(z_pre)                         # ReLU | TopK(k) | GumbelTopK(k)
y_pre = decoder(h) + pre_bias                 # decoder (tied: W_dec = W_encᵀ)
y = y_pre * σ + μ                             # undo normalization
```

Implementation details adopted for stability & performance:
- **Kaiming/He init** for encoder, **tied decoder** (decoder = encoderᵀ).
- Optional **row‑norm** of decoder (or column‑norm of encoder if tied) at init.
- **Per‑sample LN** to reduce scale sensitivity.
- Training‑side ergonomics: **AMP** (mixed precision), **grad clipping**, **cosine LR + warmup**, **pin_memory / num_workers** in loaders, and deterministic seeding.

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
  - TopK, GumbelTopK models: `NMSE` (no L1)
- **NMSE** (Normalized Mean Squared Error):
  \[ \text{NMSE} = \mathbb{E}_b\Big[ \frac{\|\hat{x}_b - x_b\|_2^2}{\|x_b\|_2^2} \Big] \]

The L1 term uses:  
\[ \text{L1} = \mathbb{E}_b\Big[ \frac{\|h_b\|_1}{\|x_b\|_2} \Big] \]

> **No auxiliary loss**: We deliberately **omit** the auxiliary reconstruction term sometimes used with Top‑K SAEs for reviving dead latents. With good init and GumbelTopK warmup, training is stable in our setting. Feel free to re‑enable an aux loss if your domain benefits from it.

---

## Training the three variants

We train (by default):
1. **ReLU+L1** (no Top‑K): `λ=1e-3` (tunable), `normalize=False`  
2. **TopK=64**: `normalize=True`, **no L1**  
3. **GumbelTopK=64**: `tau_start=1.0`, `tau_end≈1e-8`, `anneal_steps=10k`, `normalize=True`, **no L1**

For each model we collect **Loss** and **NMSE** on **train / val / test** every epoch and produce a **comparison plot** (2 rows × 3 columns; bars for each model).

---

## Quickstart

1) Put your HDF5 embeddings at `./data/per-protein.h5`  
2) Run the notebook or the provided script cell that:
   - builds dataloaders,
   - instantiates the three models,
   - trains for `EPOCHS` with Adam + cosine schedule,
   - logs/plots results.

> Hyperparameters: `n_latents`, `k`, `λ (L1)`, LR, batch size, and the Gumbel schedule (`tau_start`, `tau_end`, `anneal_steps`) are all exposed in the code.

---

## Notes on implementation details we adopted

- **Encoder init**: Kaiming uniform (good with ReLU‑like activations).  
- **Tied decoder**: fewer parameters and an inductive bias toward symmetric encode/decode.  
- **Row‑norm at init**: stabilizes early optimization (common in SAE repos).  
- **Per‑sample LN** (`layer_norm_no_affine`) instead of `nn.LayerNorm`: we explicitly **don’t learn γ/β** to keep normalization purely geometric and avoid re‑centering by the network.
- **GumbelTopK** anneals **every forward** during training, avoiding manual stepping. When `τ → τ_end`, it **reduces to hard Top‑K** automatically.
- **No auxiliary loss**: intentionally left out for simplicity; swap in if needed.

---

## Results (what to expect)

- **ReLU+L1** usually trains smoothly; sparsity depends on λ and data scale.  
- **TopK=64** yields **consistent sparsity** and often **clearer features**, but may require careful init to avoid dead units.  
- **GumbelTopK=64** tends to **match Top‑K** at convergence while **improving early training** (fewer dead units, better gradients).

Exact numbers depend on your embeddings and training budget. See the final comparison figure in the notebook for your run.

---

## References

**Core inspiration**  
- *OpenAI*: **Scaling and evaluating sparse autoencoders** — code & ideas for Top‑K training and evaluation.  
  Repo: https://github.com/openai/sparse_autoencoder

**Domain application**  
- *PNAS*: **Sparse autoencoders uncover biologically interpretable features in protein language model representations** — demonstrates biologically meaningful features in protein embeddings.  
  Repo: https://github.com/onkarsg10/Rep_SAEs_PLMs

Additional repos that influenced practical details (init, normalization, Top‑K mechanics, training ergonomics):  
- https://github.com/DanielKerrigan/saefarer  
- https://github.com/EleutherAI/sae  
- https://github.com/bartbussmann/BatchTopK  
- https://github.com/tylercosgrove/sparse-autoencoder-mistral7b  
- https://github.com/jbloomAus/SAELens  
- https://github.com/neelnanda-io/1L-Sparse-Autoencoder

---

## License

See `LICENSE` for details.
