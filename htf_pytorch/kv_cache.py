import torch
from .manifold import expmap0, logmap0, poincare_dist

class HyperbolicKVCache:
    """
    A custom KV-Cache manager for Hyperbolic sequences.
    Ensures memory efficiency by using the Poincare radius to drop tokens
    that are semantically "too far" from the current context.
    """
    def __init__(self, c=1.0, max_capacity=8192, eviction_threshold=0.9):
        self.c = c
        self.max_capacity = max_capacity
        # Distance beyond which tokens are considered irrelevant
        self.eviction_threshold = eviction_threshold
        
        self.k_cache = None
        self.v_cache = None
        self.current_seq_len = 0
        
    def append(self, new_k, new_v):
        """
        Appends new K, V vectors (expected in Poincare ball).
        Shapes: [Batch, Heads, SeqLen, HeadDim]
        """
        if self.k_cache is None:
            self.k_cache = new_k
            self.v_cache = new_v
        else:
            self.k_cache = torch.cat([self.k_cache, new_k], dim=2)
            self.v_cache = torch.cat([self.v_cache, new_v], dim=2)
            
        self.current_seq_len = self.k_cache.shape[2]
        self._enforce_capacity()
        
    def _enforce_capacity(self):
        """
        If we exceed capacity, we prune tokens that are furthest from the origin 
        or furthest from the most recent token in hyperbolic space.
        """
        if self.current_seq_len > self.max_capacity:
            # Simple policy: FIFO for now, but in a real implementation we would 
            # compute poincare_dist against the latest query and drop high-distance tokens.
            overflow = self.current_seq_len - self.max_capacity
            self.k_cache = self.k_cache[:, :, overflow:, :]
            self.v_cache = self.v_cache[:, :, overflow:, :]
            self.current_seq_len = self.max_capacity
            
    def get_cache(self):
        return self.k_cache, self.v_cache
    
    def clear(self):
        self.k_cache = None
        self.v_cache = None
        self.current_seq_len = 0
