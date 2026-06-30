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
