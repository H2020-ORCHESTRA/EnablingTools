import osmnx as ox
import networkx as nx
import igraph as ig
import os
import imageio
import pandas as pd
import math

# Initiate Globals
globals()["period"] = 60
globals()["weight"] = ["travel_car","travel_train","speed_kph","length","lanes"]

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

def adjust_for_next_period(G_ig,dict_path,routes,initial_start,edges_dict_veh,step):
    for args in dict_path.items():
        idx = args[0][1]
        edge = G_ig.es[args[0][0]]
        if args[1][1] >= (step)*period and args[1][0]<= (step)*period:
            new_start_node = edge.target
            initial_start[idx] = args[1][1] # Fix re-computing to be executed only once
            index = routes[idx].index(new_start_node)
            routes[idx] = routes[idx][index:]
            steps_fill = step-1+math.ceil((args[1][1]-(step-1)*period)/period)
            step_range = range(step,steps_fill)
            for st in step_range:
                if (args[0][0],st) not in edges_dict_veh.keys(): 
                    edges_dict_veh[args[0][0],st] = 0
                edges_dict_veh[args[0][0],st] +=1
        elif edge.target == routes[idx][-1] and args[1][1] <= (step)*period:
            routes[idx] = []
            initial_start[idx] = -1
    return routes,initial_start,edges_dict_veh

def update_dicts(G_ig,routes,initial_start,step,edges_dict_veh):
    dict_path = {}
    for idx_r,route in enumerate(routes):
        cur_time = initial_start[idx_r]
        for v in range(len(route)-1):
            edge_id = G_ig.get_eid(G_ig.vs[route[v]], G_ig.vs[route[v+1]])
            duration = min(G_ig.es[edge_id][weight[0]],G_ig.es[edge_id][weight[1]])
            dict_path[edge_id,idx_r] = (round(cur_time,2) ,round(cur_time+duration,2))
            
            if (edge_id,step) not in edges_dict_veh.keys(): 
                edges_dict_veh[edge_id,step] = 0

            if (step)*period <= round(cur_time,2) and (step+1)*period > round(cur_time,2):
                edges_dict_veh[edge_id,step]+=1

            cur_time+=duration
    edges_dict_veh = {x:y for x,y in edges_dict_veh.items() if y!=0}
    return dict_path,edges_dict_veh

def update_gdfs(G_ig,gdf_edges,edges_dict_veh,step,max_speed):

    # Update speeds of edges
    edges_passengers = [0]*len(G_ig.es)
    edges_speed_drop = [0]*len(G_ig.es)
    edges_density = [0]*len(G_ig.es)
    edges_travel_car = list(G_ig.es["travel_car"])
    edges_speed = list(G_ig.es["speed_kph"])
    gdf_edges["passengers"] = 0
    gdf_edges["density"] = 0
    gdf_edges["speed_drop"] = 0
    for edge in G_ig.es:
        index = edge.index
        if (index,step) in edges_dict_veh.keys():
            if edge['travel_train'] < edge['travel_car']:
                passengers = edges_dict_veh[index,step]
                density = round(passengers/500,2)
                speed = round(999,1)
                travel_car = edge["travel_car"]
            else:
                passengers = edges_dict_veh[index,step]
                density =  round(passengers/(edge['length']+1e-2),2)
                speed = round(max(0.1,max_speed[index]*(1-(density/(0.15*edge['lanes'])))),1)
                travel_car = edge['length']/(speed+0.01)
                gdf_edges.at[(edge.source,edge.target,0),"speed_drop"]= 1-speed/max_speed[index]
                gdf_edges.at[(edge.source,edge.target,0),"travel_car"]= travel_car     
        else:
            passengers = 0
            density = 0
            speed = round(edge["speed_kph"],1)
            travel_car = edge["travel_car"]
            gdf_edges.at[(edge.source,edge.target,0),"speed_drop"]
        gdf_edges.at[(edge.source,edge.target,0),"passengers"] = passengers
        gdf_edges.at[(edge.source,edge.target,0),"density"] = density
        gdf_edges.at[(edge.source,edge.target,0),"speed_kph"]= speed
        edges_passengers[index] = passengers
        edges_density[index] =density
        edges_speed[index] = speed

    G_ig.es['passengers'] = edges_passengers
    G_ig.es['density'] = edges_density
    G_ig.es['speed_kph'] = edges_speed
    G_ig.es['travel_car'] = edges_travel_car
    return G_ig,gdf_edges