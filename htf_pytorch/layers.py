import torch
import torch.nn as nn
from .manifold import expmap0, logmap0

class HyperbolicLinear(nn.Module):
    """
    A Hyperbolic version of nn.Linear (Mobius Linear Layer).
    It maps the hyperbolic input to the tangent space, applies standard Euclidean matrix
    multiplication, and maps the result back into hyperbolic space.
    """
    def __init__(self, in_features, out_features, c=1.0, bias=True):
        super().__init__()
        self.c = c
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # 1. Map input x from Poincare ball to tangent space at the origin
        x_tangent = logmap0(x, self.c)
        
        # 2. Apply Euclidean linear transformation
        v = self.linear(x_tangent)
        
        # 3. Map the result back into the Poincare ball
        return expmap0(v, self.c)

class HyperbolicLayerNorm(nn.Module):
    """
    Applies Layer Normalization in the tangent space and projects back.
    """
    def __init__(self, normalized_shape, c=1.0, eps=1e-5):
        super().__init__()
        self.c = c
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        x_tangent = logmap0(x, self.c)
        v = self.layer_norm(x_tangent)
        return expmap0(v, self.c)

class HyperbolicMLP(nn.Module):
    """
    Two-layer MLP operating via tangent space.
    """
    def __init__(self, embed_dim, hidden_dim, c=1.0, dropout=0.1):
        super().__init__()
        self.c = c
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_tangent = logmap0(x, self.c)
        v = self.fc1(x_tangent)
        v = self.act(v)
        v = self.dropout(v)
        v = self.fc2(v)
        v = self.dropout(v)
        return expmap0(v, self.c)

class HyperbolicTransformerBlock(nn.Module):
    """
    A full Transformer Block operating in the Poincare ball.
    Uses mobius_add for residual connections.
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, c=1.0, dropout=0.1):
        super().__init__()
        from .attention import HyperbolicMultiHeadAttention
        self.c = c
        self.ln_1 = HyperbolicLayerNorm(embed_dim, c=c)
        self.attn = HyperbolicMultiHeadAttention(num_heads, embed_dim, c=c)
        self.ln_2 = HyperbolicLayerNorm(embed_dim, c=c)
        self.mlp = HyperbolicMLP(embed_dim, hidden_dim, c=c, dropout=dropout)

    def forward(self, x):
        from .manifold import mobius_add
        
        # Pre-LN Attention
        norm_x = self.ln_1(x)
        # We project norm_x to Q, K, V within the attention module if needed, 
        # or we just use norm_x for all three (self-attention).
        # In a real implementation we would have HyperbolicLinear layers for Q, K, V.
        # For simplicity, we just use norm_x directly, but let's add the linear projections.
        
        # Actually, self-attention needs Q, K, V projections.
        # Let's do that inline or assume the attention module takes care of it.
        # Standard PyTorch nn.MultiheadAttention includes the projections.
        # Since our HyperbolicMultiHeadAttention doesn't have linear projections yet, 
        # let's just pass norm_x as Q, K, V.
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = mobius_add(x, attn_out, self.c)
        
        # Pre-LN MLP
        norm_x2 = self.ln_2(x)
        mlp_out = self.mlp(norm_x2)
        x = mobius_add(x, mlp_out, self.c)
        
        return x

