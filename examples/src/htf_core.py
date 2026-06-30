import numpy as np

def embed_hyperbolic(tree):
    """
    Embeds a tree into the Poincaré disk (Hyperbolic Space). 
    The root is placed at the origin. Children branch outward concentrically.
    As depth increases, nodes pack exponentially tighter near the edge, 
    but their mathematical distance grows appropriately.
    """
    pos = {}
    root = 0
    pos[root] = np.array([0.0, 0.0])
    
    # BFS to assign positions
    # queue tuple: (node, depth, angle_start, angle_end)
    queue = [(root, 0, 0, 2 * np.pi)] 
    
    while queue:
        node, depth, a_start, a_end = queue.pop(0)
        
        # Handle both directed and undirected networkx graphs cleanly
        if tree.is_directed():
            children = list(tree.successors(node))
        else:
            children = [n for n in tree.neighbors(node) if n not in pos]
        
        if not children:
            continue
            
        angle_step = (a_end - a_start) / len(children)
        # The radius approaches 1 (the boundary of the Poincaré disk) as depth increases
        r = 1.0 - (1.0 / (depth + 2)) 
        
        for i, child in enumerate(children):
            child_angle = a_start + i * angle_step + angle_step / 2
            x = r * np.cos(child_angle)
            y = r * np.sin(child_angle)
            pos[child] = np.array([x, y])
            queue.append((child, depth + 1, a_start + i * angle_step, a_start + (i + 1) * angle_step))
            
    return pos

def poincare_distance(u, v):
    """
    Calculates the exact Hyperbolic distance between two vectors inside the Poincaré disk.
    Formula: d(u,v) = arcosh(1 + 2 * ||u-v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))
    """
    u_norm_sq = np.sum(u**2)
    v_norm_sq = np.sum(v**2)
    diff_sq = np.sum((u - v)**2)
    
    # Prevent division by zero near the edge (numerical stability constraint)
    u_norm_sq = min(u_norm_sq, 1 - 1e-7)
    v_norm_sq = min(v_norm_sq, 1 - 1e-7)
    
    gamma = 1 + 2 * diff_sq / ((1 - u_norm_sq) * (1 - v_norm_sq))
    return np.arccosh(gamma)
