import os
import pickle
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from networkx.algorithms import community


GRAPH_PATH = "data/pyg_graphs/graphs.pk_fix_short"
PLOT_DIR = "data/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

with open(GRAPH_PATH, "rb") as f:
    graphs = pickle.load(f)

print(f"{len(graphs)} graphs loaded.")

label_names = {0: "CN", 2: "AD"}
label_colors = {0: "blue", 1: "orange", 2: "purple"}

N = 14

for i, data in enumerate(graphs[:N]):
    G = to_networkx(data, to_undirected=True)
    label = data.y.item()
    label_name = label_names[label]
    color = label_colors[label]

    pos = nx.spring_layout(G, seed=42)

    try:
        communities = community.greedy_modularity_communities(G)
    except:
        communities = [list(G.nodes)]

    isolated_nodes = list(nx.isolates(G))

    degrees = dict(G.degree())
    node_sizes = [max(degrees[n], 1) * 10 for n in G.nodes()]

    plt.figure(figsize=(7, 7))
    for j, com in enumerate(communities):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(com),
            node_color=color,
            node_size=[node_sizes[n] for n in com],
            alpha=0.8,
            label=f"Community {j+1} ({len(com)} nodes)"
        )

    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.4)

    if isolated_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=isolated_nodes,
            node_color="pink",
            node_size=60,
            label="Isolated nodes"
        )

    plt.title(f"Graph #{i+1} – {label_name} ({label}) | Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}",
              fontsize=10)
    plt.legend(fontsize=8, loc="upper right", frameon=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"graph_{i+1}.png"))
    plt.close()


    density = nx.density(G)
    num_components = nx.number_connected_components(G)
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    print(f"Graph #{i+1}:")
    print(f"   Label: {label_name} ({label})")
    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"   Isolated nodes: {len(isolated_nodes)}")
    print(f"   Connected components: {num_components}")
    print(f"   Density: {density:.4f}, Mean degree: {avg_degree:.2f}")
    print("—" * 40)
