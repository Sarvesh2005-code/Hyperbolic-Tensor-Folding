import torch

def _clamp_norm(x, c, p=2, dim=-1, keepdim=True):
    """
    Clamps the norm of the tensor strictly below the radius of the Poincare ball
    to maintain numerical stability and prevent NaNs.
    Radius = 1 / sqrt(c)
    """
    max_norm = (1.0 / (c ** 0.5)) - 1e-5
    norm = torch.norm(x, p=p, dim=dim, keepdim=keepdim)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)

def mobius_add(x, y, c=1.0):
    """
    Mobius addition in the Poincare ball of curvature c.
    Formula: x (+) y = ((1 + 2c<x,y> + c|y|^2)x + (1 - c|x|^2)y) / (1 + 2c<x,y> + c^2|x|^2|y|^2)
    """
    x = _clamp_norm(x, c)
    y = _clamp_norm(y, c)

    cx2 = c * torch.sum(x * x, dim=-1, keepdim=True)
    cy2 = c * torch.sum(y * y, dim=-1, keepdim=True)
    cxy = c * torch.sum(x * y, dim=-1, keepdim=True)

    num = (1 + 2 * cxy + cy2) * x + (1 - cx2) * y
    den = 1 + 2 * cxy + cx2 * cy2

    res = num / den.clamp_min(1e-15)
    return _clamp_norm(res, c)

def expmap0(v, c=1.0):
    """
    Exponential map from the tangent space at the origin to the Poincare ball.
    Moves standard Euclidean vectors into Hyperbolic space.
    """
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    sqrt_c = c ** 0.5
    res = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
    return _clamp_norm(res, c)

def logmap0(y, c=1.0):
    """
    Logarithmic map from the Poincare ball back to the tangent space at the origin.
    Moves Hyperbolic vectors back to standard Euclidean space.
    """
    y = _clamp_norm(y, c)
    y_norm = torch.norm(y, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    sqrt_c = c ** 0.5
    return torch.atanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm)

def poincare_dist(x, y, c=1.0):
    """
    Calculates the exact hyperbolic distance between two points in the Poincare ball.
    """
    sqrt_c = c ** 0.5
    diff = mobius_add(-x, y, c)
    diff_norm = torch.norm(diff, p=2, dim=-1).clamp_min(1e-15)
    dist = 2 / sqrt_c * torch.atanh((sqrt_c * diff_norm).clamp_max(1 - 1e-5))
    return dist
