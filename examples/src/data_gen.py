import networkx as nx

def generate_semantic_tree(branching_factor=3, depth=4):
    """
    Generates a hierarchical tree representing AI semantic logic.
    For example, an ontology of words where concepts break down into sub-concepts.
    """
    tree = nx.balanced_tree(branching_factor, depth)
    return tree
