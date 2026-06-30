# Hyperbolic Tensor Folding (HTF) for PyTorch

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

HTF is a revolutionary mathematical architecture for Large Language Models (LLMs) that maps attention mechanisms and KV-Caches into non-Euclidean hyperbolic space (the Poincaré disk).

Because human language and semantic structures are inherently hierarchical (like a tree), traditional flat-space (Euclidean) matrix multiplication forces the KV-Cache to grow polynomially, leading to massive RAM constraints and "out of memory" errors on consumer hardware.

**Hyperbolic Tensor Folding** mathematically folds this context into the Poincaré disk, allowing near-infinite hierarchical semantic bounds inside a finite coordinate radius, exponentially shrinking memory requirements.

## Installation

```bash
pip install .
```

## Usage

Replace your standard Linear and Attention modules with their Hyperbolic equivalents!

```python
import torch
from htf_pytorch import HyperbolicLinear, HyperbolicAttention

# Initialize our Hyperbolic Attention Layer
attention = HyperbolicAttention(c=1.0) # c = curvature

# Create dummy Query, Key, Value tensors in the Poincaré ball
q = torch.rand(2, 8, 64) * 0.1
k = torch.rand(2, 16, 64) * 0.1
v = torch.rand(2, 16, 64) * 0.1

# Run the forward pass!
output, weights = attention(q, k, v)
print(f"Output shape: {output.shape}")
```

## License
This project is licensed under the **Apache 2.0 License** - see the [LICENSE](LICENSE) file for details.
