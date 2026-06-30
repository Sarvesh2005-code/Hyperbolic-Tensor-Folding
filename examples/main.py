import sys
from src.data_gen import generate_semantic_tree
from src.euclidean_baseline import embed_euclidean, calculate_euclidean_distance
from src.htf_core import embed_hyperbolic, poincare_distance
from src.visualizer import plot_embeddings

def main():
    print("=" * 60)
    print("   HYPERBOLIC TENSOR FOLDING (HTF) - RESEARCH SIMULATOR")
    print("=" * 60)
    
    # 1. Generate Data
    print("[*] Generating synthetic hierarchical semantic tree...")
    # Branching factor 3, depth 5 = 364 nodes. A good size for visualization.
    tree = generate_semantic_tree(branching_factor=3, depth=5)
    print(f"[-] Tree generated with {len(tree.nodes)} interconnected nodes (simulating AI tokens).")
    
    # 2. Euclidean Baseline
    print("\n[*] Calculating Standard Euclidean Embeddings (Current LLM Standard)...")
    eucl_pos = embed_euclidean(tree)
    
    # 3. Hyperbolic Embedding
    print("[*] Calculating Hyperbolic Embeddings (Poincaré Disk)...")
    hyp_pos = embed_hyperbolic(tree)
    
    # 4. Benchmarking
    print("\n[*] Benchmarking Mathematical Distance Properties...")
    root = 0
    leaf = max(tree.nodes) # Grab the furthest generated leaf node
    
    eucl_dist = calculate_euclidean_distance(eucl_pos[root], eucl_pos[leaf])
    hyp_dist = poincare_distance(hyp_pos[root], hyp_pos[leaf])
    
    print(f"    -> Euclidean Distance (Root to Leaf): {eucl_dist:.4f}")
    print(f"    -> Hyperbolic Distance (Root to Leaf): {hyp_dist:.4f}")
    print("\n[!] INSIGHT: Notice how the hyperbolic distance can grow infinitely large")
    print("    even though the physical coordinates never exceed a radius of 1.")
    print("    This allows us to 'fold' massive context windows into tiny RAM footprints.")
    
    # 5. Visualization
    print("\n[*] Generating Mathematical Visualizations...")
    plot_embeddings(tree, eucl_pos, hyp_pos, output_dir="output")
    print("=" * 60)

if __name__ == "__main__":
    main()
