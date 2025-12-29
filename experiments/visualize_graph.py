import json
import networkx as nx
import matplotlib.pyplot as plt
import os

def visualize_graph(json_path, output_path):
    print(f"Loading graph from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create graph
    G = nx.MultiDiGraph() if data.get('multigraph') else nx.DiGraph()

    # Add nodes
    print(f"Adding {len(data['nodes'])} nodes...")
    for node in data['nodes']:
        G.add_node(node['id'], **node)

    # Add edges
    print(f"Adding {len(data['edges'])} edges...")
    for edge in data['edges']:
        # NetworkX expects u, v, key, attr_dict
        # The JSON has source, target, key, and other attributes
        u = edge.pop('source')
        v = edge.pop('target')
        k = edge.pop('key', None)
        G.add_edge(u, v, key=k, **edge)

    print("Generating layout...")
    plt.figure(figsize=(20, 20))
    
    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Draw nodes
    # Color nodes by type if available
    node_colors = []
    for node in G.nodes(data=True):
        n_type = node[1].get('type', 'Unknown')
        if n_type == 'Entity':
            node_colors.append('lightblue')
        else:
            node_colors.append('lightgray')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    # Draw edge labels (optional, might be too cluttered)
    # edge_labels = {(u, v): d.get('relation', '')[:20] for u, v, d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title(f"Graph Visualization: {os.path.basename(json_path)}")
    plt.axis('off')
    
    print(f"Saving visualization to {output_path}...")
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    print("Done!")

if __name__ == "__main__":
    base_dir = "/home/ubuntu/hzy/crl/Amadeus/amadeus/experiments/data"
    json_file = os.path.join(base_dir, "graph_conv-26.json")
    output_file = os.path.join(base_dir, "graph_conv-26.png")
    
    visualize_graph(json_file, output_file)
