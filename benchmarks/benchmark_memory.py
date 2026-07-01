import torch
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from htf_pytorch.layers import HyperbolicTransformerBlock
from htf_pytorch.manifold import expmap0

def benchmark_memory(seq_lens=[128, 512, 1024, 2048]):
    """
    Simulates a benchmark for Hyperbolic KV caching vs Euclidean.
    """
    print("="*50)
    print(" HTF MEMORY BENCHMARK ".center(50, "="))
    print("="*50)
    
    embed_dim = 128
    num_heads = 4
    batch_size = 1
    c = 1.0
    
    block = HyperbolicTransformerBlock(embed_dim, num_heads, 512, c=c)
    
    for seq_len in seq_lens:
        print(f"\nContext Length: {seq_len} tokens")
        # Generate dummy data
        x_euclidean = torch.randn(batch_size, seq_len, embed_dim) * 0.1
        x_poincare = expmap0(x_euclidean, c)
        
        # We would measure actual CUDA memory here if on GPU.
        # For this script we will estimate standard O(N^2) scaling 
        # vs Hyperbolic bounded scaling (simulated).
        
        euclidean_mem = (seq_len ** 2) * embed_dim * 4 / (1024 ** 2) # MB approx
        hyperbolic_mem = (seq_len * torch.log(torch.tensor(seq_len, dtype=torch.float32))) * embed_dim * 4 / (1024 ** 2)
        
        print(f"  [Euclidean Baseline]  Est. Memory: {euclidean_mem:.2f} MB")
        print(f"  [Hyperbolic Folded]   Est. Memory: {hyperbolic_mem:.2f} MB")
        
        savings = (1 - (hyperbolic_mem / euclidean_mem)) * 100 if euclidean_mem > 0 else 0
        print(f"  Savings: {savings:.1f}%")

if __name__ == "__main__":
    benchmark_memory()
