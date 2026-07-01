import torch
import pytest
from htf_pytorch.layers import HyperbolicTransformerBlock
from htf_pytorch.manifold import expmap0

def test_transformer_block_forward():
    embed_dim = 64
    num_heads = 2
    hidden_dim = 128
    c = 1.0
    
    block = HyperbolicTransformerBlock(
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        hidden_dim=hidden_dim, 
        c=c
    )
    
    batch_size = 2
    seq_len = 8
    
    # Random Euclidean data
    x_euclidean = torch.randn(batch_size, seq_len, embed_dim) * 0.1
    
    # Map to Poincare ball
    x_poincare = expmap0(x_euclidean, c)
    
    # Ensure it's inside the ball initially
    max_input_norm = torch.max(torch.norm(x_poincare, dim=-1))
    assert max_input_norm < (1.0 / (c ** 0.5)), "Input exceeded Poincaré ball radius"
    
    # Forward pass
    out_poincare = block(x_poincare)
    
    # Check shape
    assert out_poincare.shape == (batch_size, seq_len, embed_dim)
    
    # Check output is strictly inside the Poincare ball
    max_out_norm = torch.max(torch.norm(out_poincare, dim=-1))
    assert max_out_norm <= (1.0 / (c ** 0.5)), "Output exceeded Poincaré ball radius"
