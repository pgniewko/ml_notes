# Magnitude-Dependent and Adaptive Dropout in Deep Learning

This document collects key references and implementations of dropout variants where the probability of a connection or neuron being dropped depends on magnitude, importance, or is adaptively learned.

---

## 📑 Key Papers & Concepts

| Title / Authors | Summary / What they do | Relation to “magnitude‑dependent dropout” / adaptivity |
|---|---|---|
| **Adaptive Dropout for Training Deep Neural Networks** (Ba & Frey, NeurIPS 2013) | Uses a belief network to *learn* dropout probabilities for hidden units, conditioning them on activations. [PDF](https://proceedings.neurips.cc/paper/2013/file/7b5b23f4aadf9513306bcd59afb6e4c9-Paper.pdf?utm_source=chatgpt.com) | Learned dropout probabilities correlate with neuron magnitude/importance. |
| **Variational Dropout and the Local Reparameterization Trick** (Kingma, Salimans, Welling, NeurIPS 2015) | Reinterprets dropout as variational inference, learning dropout rates instead of fixing them. [PDF](https://proceedings.neurips.cc/paper/2015/file/bc7316929fe1545bf0b98d114ee3ecb8-Paper.pdf?utm_source=chatgpt.com) | Dropout rates become parameters; weaker weights often get higher dropout. |
| **Variational Dropout Sparsifies Deep Neural Networks** (Molchanov et al., ICML 2017) | Extends variational dropout with per-weight rates, leading to sparse solutions. [arXiv](https://arxiv.org/abs/1701.05369?utm_source=chatgpt.com) | Directly realizes magnitude-driven dropout; small weights → high dropout. |
| **Adaptive Network Sparsification with Dependent Variational Beta‑Bernoulli Dropout** (Lee et al., 2018) | Introduces Beta-Bernoulli priors for adaptive, input-dependent dropout. [arXiv](https://arxiv.org/abs/1805.10896?utm_source=chatgpt.com) | Input/context-dependent dropout decisions. |
| **Contextual Dropout: An Efficient Sample‑Dependent Dropout Module** (Fan et al., 2021) | Dropout probabilities per sample via a lightweight module. [arXiv](https://arxiv.org/abs/2103.04181?utm_source=chatgpt.com) | Learns sample/context-based dropout probabilities. |
| **Variational Bayesian Dropout With a Hierarchical Prior** (Liu et al., CVPR 2019) | Introduces hierarchical priors to stabilize variational dropout. [PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Variational_Bayesian_Dropout_With_a_Hierarchical_Prior_CVPR_2019_paper.pdf?utm_source=chatgpt.com) | Provides Bayesian grounding for magnitude-based dropout. |
| **Advanced Dropout: A Model‑free Methodology for Bayesian Dropout Optimization** (2020) | Defines a parametric prior over dropout masks, learned via stochastic variational Bayes. [arXiv](https://arxiv.org/abs/2010.05244?utm_source=chatgpt.com) | Adaptively adjusts dropout rates in a Bayesian framework. |

---

## 💻 Implementations / Code Repositories

- [bayesgroup / variational-dropout-sparsifies-dnn (TensorFlow)](https://github.com/bayesgroup/variational-dropout-sparsifies-dnn?utm_source=chatgpt.com) – Sparse Variational Dropout implementation.
- [kefirski / variational_dropout (PyTorch)](https://github.com/kefirski/variational_dropout?utm_source=chatgpt.com) – PyTorch implementation of variational dropout.
- [elliothe / Variational_dropout (PyTorch)](https://github.com/elliothe/Variational_dropout?utm_source=chatgpt.com) – PyTorch version with fixes.
- [xuhang07 / Adpative-Dropout](https://github.com/xuhang07/Adpative-Dropout?utm_source=chatgpt.com) – Implementation for adaptive dropout across layers.
- [AdvancedDropout Codebase](https://arxiv.org/abs/2010.05244?utm_source=chatgpt.com) – Code provided with the Advanced Dropout paper.

---

## ⚠️ Notes & Caveats

- Learning dropout rates per weight/neuron can cause **training instability** (especially for extreme rates).  
- Sparse variational dropout requires reparameterization tricks to stabilize gradients.  
- Some adaptive methods add significant overhead (e.g. auxiliary networks for learning probabilities).  
- Naive magnitude-based dropout can lead to **over-pruning**; careful priors or hierarchical models mitigate this.  

---

## ✅ Suggested Starting Points

- For practical use: try **Variational Dropout Sparsifies DNNs (Molchanov et al. 2017)** – robust and well-studied.  
- For more modern takes: experiment with **Contextual Dropout (Fan et al. 2021)** or **Advanced Dropout (2020)**.  
- Use adaptive methods if you want input-dependent regularization.

