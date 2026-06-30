import matplotlib.pyplot as plt
import networkx as nx
import os

def plot_embeddings(tree, eucl_pos, hyp_pos, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("AI Memory Compression via Hyperbolic Geometry", fontsize=20, fontweight='bold')
    
    # --- Euclidean Plot ---
    ax = axes[0]
    ax.set_title("Standard Euclidean KV-Cache (Flat Space)\nRequires massive dimensions to prevent overlap.", fontsize=12)
    nx.draw_networkx_edges(tree, eucl_pos, ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(tree, eucl_pos, ax=ax, node_size=25, node_color='#4A90E2', alpha=0.8)
    ax.axis('equal')
    ax.axis('off')
    
    # --- Hyperbolic Plot ---
    ax = axes[1]
    ax.set_title("Hyperbolic Tensor Folding (Poincaré Disk)\nInfinite hierarchical capacity bound within a unit circle.", fontsize=12)
    
    # Draw the unit circle boundary (the edge of the universe in hyperbolic space)
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5)
    ax.add_artist(circle)
    
    nx.draw_networkx_edges(tree, hyp_pos, ax=ax, alpha=0.3, edge_color='gray')
    nx.draw_networkx_nodes(tree, hyp_pos, ax=ax, node_size=25, node_color='#D0021B', alpha=0.8)
    
    ax.axis('equal')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
    save_path = os.path.join(output_dir, "htf_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[+] Success: High-resolution visualization saved to {save_path}")
    
    # We will just print the path, user can open it.
