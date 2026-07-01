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

class HyperbolicMultiHeadAttention(nn.Module):
    """
    Multi-Head Hyperbolic Attention Mechanism.
    Splits the embedding dimension into multiple heads, computes hyperbolic distances,
    and recombines the heads.
    """
    def __init__(self, num_heads, embed_dim, c=1.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.c = c
        
    def forward(self, q, k, v):
        """
        Expects q, k, v inside the Poincare ball.
        Shapes: [Batch, SeqLen, EmbedDim]
        """
        B, L_Q, _ = q.size()
        _, L_K, _ = k.size()
        
        # Reshape to [Batch, SeqLen, Heads, HeadDim] and then [Batch, Heads, SeqLen, HeadDim]
        # In tangent space / euclidean space, reshaping doesn't change the underlying vector, 
        # but in hyperbolic space we must ensure norms stay within the ball. Since we are just
        # taking slices of the coordinates, if the total norm is < 1/sqrt(c), the slice norm 
        # is also < 1/sqrt(c). 
        q = q.view(B, L_Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_K, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_K, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Expand for pairwise dist
        # q_exp: [Batch, Heads, L_Q, 1, HeadDim]
        q_exp = q.unsqueeze(3)
        # k_exp: [Batch, Heads, 1, L_K, HeadDim]
        k_exp = k.unsqueeze(2)
        
        # Calculate distance
        dist = poincare_dist(q_exp, k_exp, self.c)
        
        # Attention weights
        # Scaling by sqrt(head_dim) is sometimes used in Euclidean to prevent vanishing gradients,
        # but in hyperbolic space, distances naturally scale differently. We can use a learned scalar or just standard softmax.
        attn_weights = torch.softmax(-dist, dim=-1)
        
        # Average in tangent space
        v_tangent = logmap0(v, self.c)
        # v_tangent: [Batch, Heads, L_K, HeadDim]
        # attn_weights: [Batch, Heads, L_Q, L_K]
        # out_tangent: [Batch, Heads, L_Q, HeadDim]
        out_tangent = torch.matmul(attn_weights, v_tangent)
        
        # Map back to Poincare
        out_poincare = expmap0(out_tangent, self.c)
        
        # Reshape back to [Batch, L_Q, EmbedDim]
        out_poincare = out_poincare.transpose(1, 2).contiguous().view(B, L_Q, -1)
        
        # NOTE: After concatenation, the norm could potentially exceed the Poincare bound 
        # if the individual head vectors were near the boundary. However, our manifold functions 
        # clamp norms defensively. To be perfectly mathematically sound, we should project the 
        # concatenated result.
        from .manifold import _clamp_norm
        out_poincare = _clamp_norm(out_poincare, self.c)
        
        return out_poincare, attn_weights

