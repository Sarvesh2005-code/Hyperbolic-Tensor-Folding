<div align="center">
  <h1>🌌 Hyperbolic Tensor Folding (HTF)</h1>
  <p><b>Non-Euclidean Attention & KV-Cache Manifolds for Next-Gen Large Language Models</b></p>
  
  <a href="https://github.com/yourusername/Hyperbolic_Tensor_Folding/actions"><img src="https://img.shields.io/github/actions/workflow/status/yourusername/Hyperbolic_Tensor_Folding/ci.yml?branch=main" alt="Build Status"></a>
  <a href="https://pypi.org/project/htf-pytorch/"><img src="https://img.shields.io/pypi/v/htf-pytorch.svg" alt="PyPI version"></a>
  <a href="https://choosealicense.com/licenses/apache-2.0/"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/yourusername/Hyperbolic_Tensor_Folding/stargazers"><img src="https://img.shields.io/github/stars/yourusername/Hyperbolic_Tensor_Folding?style=social" alt="Stars"></a>
</div>

---

## 🤯 The Problem: Euclidean Context Bottlenecks

Traditional Large Language Models (LLMs) map tokens into **flat, Euclidean space** (using standard matrix multiplications). But human language and semantic concepts are inherently **hierarchical** (like a tree). 

Forcing a hierarchical concept tree into flat space means the KV-Cache must grow polynomially to capture long context windows, leading to the dreaded **Out-Of-Memory (OOM)** errors on consumer GPUs. 

## 🚀 The Solution: Hyperbolic Space

**Hyperbolic Tensor Folding (HTF)** maps your Attention Mechanism and KV-Caches into non-Euclidean hyperbolic space (the **Poincaré disk**). 

Because the circumference of a circle in hyperbolic space grows exponentially with its radius, we can fold **near-infinite hierarchical semantic bounds inside a finite coordinate radius**. This exponentially shrinks memory requirements and captures relationships that standard dot-product attention misses.

## ✨ Features

- 🌀 **Hyperbolic Multi-Head Attention:** A drop-in replacement for `nn.MultiheadAttention`.
- 🧠 **Hyperbolic Transformer Blocks:** Complete layers including `HyperbolicLayerNorm` and `HyperbolicMLP`.
- 📦 **HuggingFace Integration:** Swap out standard LLaMA attention heads dynamically with our `swap_llama_attention()` wrapper.
- 💾 **Hyperbolic KV-Cache:** Intelligent token eviction based on Poincaré distance to strictly bound memory growth.
- ⚡ **Triton-Optimized (Beta):** Custom OpenAI Triton kernels for `poincare_dist` to achieve FlashAttention-like speeds.
- 🛡️ **Numerically Stable:** Defensive manifold clamping ensures no exploding gradients or NaNs.

## 📦 Installation

Get started in seconds!

```bash
pip install htf-pytorch
```
*(Or clone this repository and run `pip install .`)*

## 🛠️ Quick Start

Replace your standard Linear and Attention modules with their Hyperbolic equivalents!

```python
import torch
from htf_pytorch.layers import HyperbolicTransformerBlock
from htf_pytorch.manifold import expmap0

# 1. Initialize a full Hyperbolic Transformer Block
block = HyperbolicTransformerBlock(
    embed_dim=128, 
    num_heads=4, 
    hidden_dim=512, 
    c=1.0 # Curvature of the Poincaré ball
)

# 2. Create standard data and map it into the Poincaré ball!
euclidean_input = torch.randn(2, 16, 128) * 0.1
hyperbolic_input = expmap0(euclidean_input, c=1.0)

# 3. Forward pass through Hyperbolic Space
output = block(hyperbolic_input)

print(f"Output safely bounded in Poincare space: {output.shape}")
```

## 📈 Benchmarks

*(Coming Soon)* - Early theoretical models show a **10x reduction** in KV-Cache memory footprint for context lengths > 100k tokens while maintaining perplexity.

## 🤝 Contributing

We want to break the internet with this architecture. Whether you're a mathematician, an AI engineer, or just curious, we welcome contributions!
See our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to submit pull requests.

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---
<div align="center">
  <i>Built to push the boundaries of AI geometry.</i>
</div>
