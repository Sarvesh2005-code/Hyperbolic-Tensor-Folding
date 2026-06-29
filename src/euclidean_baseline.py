import networkx as nx
import numpy as np

def embed_euclidean(tree):
    """
    Embeds the tree using a standard Euclidean force-directed layout.
    This mimics how current standard KV-Caches store token embeddings in flat space.
    """
    pos = nx.spring_layout(tree, seed=42, dim=2)
    return pos
    
def calculate_euclidean_distance(u, v):
    """Standard L2 Norm (Euclidean Distance)."""
    return np.linalg.norm(np.array(u) - np.array(v))
