import torch
import sys
import os

# Add the parent directory to the path so we can import htf_pytorch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from htf_pytorch.layers import HyperbolicTransformerBlock
from htf_pytorch.manifold import expmap0

def main():
    print("Initializing Hyperbolic Transformer Block...")
    embed_dim = 128
    num_heads = 4
    hidden_dim = 512
    c = 1.0
    
    block = HyperbolicTransformerBlock(
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        hidden_dim=hidden_dim, 
        c=c
    )
    
    # Create dummy data in Euclidean space and map to Poincare ball
    batch_size = 2
    seq_len = 16
    x_euclidean = torch.randn(batch_size, seq_len, embed_dim) * 0.1
    x_poincare = expmap0(x_euclidean, c)
    
    print(f"Input shape: {x_poincare.shape}")
    print(f"Max input norm: {torch.max(torch.norm(x_poincare, dim=-1))}")
    
    # Forward pass
    print("Running forward pass...")
    out_poincare = block(x_poincare)
    
    print(f"Output shape: {out_poincare.shape}")
    
    max_out_norm = torch.max(torch.norm(out_poincare, dim=-1))
    print(f"Max output norm: {max_out_norm}")
    
    max_allowed_norm = (1.0 / (c ** 0.5)) - 1e-5
    print(f"Max allowed norm (radius): {max_allowed_norm}")
    
    if max_out_norm <= max_allowed_norm + 1e-4:
        print("SUCCESS! Output remained within the Poincaré ball boundaries.")
    else:
        print("ERROR! Norm exceeded boundaries.")

if __name__ == "__main__":
    main()
