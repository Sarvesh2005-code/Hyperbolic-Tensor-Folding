import torch
import torch.nn as nn
from .manifold import poincare_dist, expmap0, logmap0

class HyperbolicAttention(nn.Module):
    """
    Hyperbolic Attention Mechanism for LLMs.
    Replaces standard dot-product attention with Negative Hyperbolic Distance calculation.
    """
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c

    def forward(self, q, k, v):
        """
        Expects queries (q), keys (k), and values (v) inside the Poincare ball.
        Shapes:
            q: [Batch, SeqLen_Q, Dim]
            k: [Batch, SeqLen_K, Dim]
            v: [Batch, SeqLen_K, Dim]
        """
        # Expand dimensions to broadcast pairwise distances
        # q_exp: [Batch, SeqLen_Q, 1, Dim]
        q_exp = q.unsqueeze(2)
        # k_exp: [Batch, 1, SeqLen_K, Dim]
        k_exp = k.unsqueeze(1)
        
        # Calculate Hyperbolic Distance between all Q and K
        # dist: [Batch, SeqLen_Q, SeqLen_K]
        dist = poincare_dist(q_exp, k_exp, self.c)
        
        # Calculate attention weights. Close points (dist -> 0) get higher score.
        # Negative distance acts inversely to dot-product similarities.
        attn_weights = torch.softmax(-dist, dim=-1)
        
        # To strictly average values in Hyperbolic space, we use the Einstein Midpoint.
        # For production speed, we approximate by averaging in tangent space (LogEuclidean).
        v_tangent = logmap0(v, self.c)
        
        # Weighted sum of values in tangent space
        out_tangent = torch.matmul(attn_weights, v_tangent)
        
        # Map back to Poincare ball
        output = expmap0(out_tangent, self.c)
        
        return output, attn_weights
