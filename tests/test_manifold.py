import torch
import pytest
from htf_pytorch.manifold import mobius_add, expmap0, logmap0, poincare_dist

def test_exp_log_map():
    # Test that log(exp(x)) == x
    v = torch.tensor([0.1, 0.2, 0.3])
    hyp = expmap0(v)
    v_back = logmap0(hyp)
    assert torch.allclose(v, v_back, atol=1e-5), "Exp and Log maps are not inverses!"

def test_poincare_distance_origin():
    # Distance from origin should just be exactly related to the norm
    x = torch.tensor([0.0, 0.0])
    y = torch.tensor([0.5, 0.0])
    dist = poincare_dist(x, y)
    
    # Mathematical known distance from origin to (0.5, 0)
    expected_dist = 2 * torch.atanh(torch.tensor(0.5))
    assert torch.allclose(dist, expected_dist, atol=1e-5)

def test_boundary_stability():
    # Test numerical stability when tensors approach the edge of the Poincare ball
    x = torch.tensor([0.99999, 0.0])
    y = torch.tensor([-0.99999, 0.0])
    
    dist = poincare_dist(x, y)
    assert not torch.isnan(dist), "Boundary distance resulted in NaN!"
    
    res = mobius_add(x, y)
    assert not torch.isnan(res).any(), "Mobius addition near boundary resulted in NaN!"
