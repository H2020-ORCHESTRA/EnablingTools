import osmnx as ox
import networkx as nx
import igraph as ig
import os
import imageio
import pandas as pd
from  matplotlib.colors import LinearSegmentedColormap

# Initiate Globals
globals()["weight"] = ["travel_car","travel_train","speed_kph","length","lanes"]
cmap=LinearSegmentedColormap.from_list('rg',["g", "w","y","r"], N=256) 

def transform_nx_to_igragh(G):
    osmids = list(G.nodes)
    G = nx.relabel.convert_node_labels_to_integers(G)
    G_ig = ig.Graph(directed=True)
    G_ig.add_vertices(G.nodes)
    G_ig.add_edges(G.edges())
    G_ig.vs["osmid"] = osmids
    for w in weight:
        weights = list((nx.get_edge_attributes(G, w).values()))
        try:        
            weights = [int(float(w)) for w in weights]
        except:
            weights = [int(max(w)) for w in weights]
        G_ig.es[w] = weights
        osmid_values = {k: v for k, v in zip(G.nodes, osmids)}
        nx.set_node_attributes(G, osmid_values, "osmid")
        assert len(G.nodes()) == G_ig.vcount()
        assert len(G.edges()) == G_ig.ecount()
    return G,G_ig

def get_max_speed(G_ig):
    max_speed = {}
    for edge in G_ig.es:
        max_speed[edge.index] = edge["speed_kph"]
    return max_speed

def load_train_schedule(name="train_schedule.xlsx"):
    train_dir = os.path.join(os.getcwd(),"core\\train")
    try:
        df = pd.read_excel(os.path.join(train_dir,name))
        df.set_index(["u_ig"], inplace = True)
        return df
    except:
        pass

def plot_map(gdf_nodes,gdf_edges,area):

    # Break down graph to nodes and edges
    G = ox.graph_from_gdfs(gdf_nodes, gdf_edges)
    edges = (gdf_edges[gdf_edges['mode'] =="train"].index).tolist()
    nodes_o = [x[0] for x in edges]
    nodes_d = [x[1] for x in edges]
    nodes = list(set(nodes_o) | set(nodes_d))

    # Set plot Values
    ec = ['y' if ((u,v,k) in edges) else 'b' for u, v, k in G.edges(keys=True)]
    el = [3 if ((u,v,k) in edges) else 0.5 for u, v, k in G.edges(keys=True)]
    nc = ["g" if (u in nodes) else "b" for u in G.nodes]
    ns = [30 if (u in nodes) else 0.15 for u in G.nodes]

    # Plot Map
    ox.plot_graph(G,bgcolor = "w",edge_color = ec,edge_linewidth=el,node_color=nc,node_size=ns,figsize = (50,32),dpi = 300,save=True)

def save_to_gif():
    cwd = os.getcwd()
    png_dir = 'core\pics'
    images = []
    for file_name in sorted(os.listdir(os.path.join(cwd,png_dir))):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(os.path.join(cwd,png_dir,'movie.gif'), images,fps = 1)