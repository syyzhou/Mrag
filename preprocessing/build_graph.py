import os
import pickle
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def build_global_POI_checkin_graph(df, exclude_user=None):
    G = nx.DiGraph()
    users = list(set(df['UserId'].to_list()))
    if exclude_user in users: users.remove(exclude_user)
    loop = tqdm(users)

    for user_id in loop:
        user_df = df[df['UserId'] == user_id]

        # Add nodes (POIs)
        for _, row in user_df.iterrows():
            node = row['PoiId']
            if node not in G.nodes():
                G.add_node(node,
                           checkin_cnt=1,
                           latitude=row['Latitude'],
                           longitude=row['Longitude'])
            else:
                G.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in sequence)
        previous_poi_id = 0
        previous_traj_id = 0
        for _, row in user_df.iterrows():
            poi_id = row['PoiId']
            traj_id = row['trajectory_id']
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue

            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:
                G.add_edge(previous_poi_id, poi_id, weight=1)

            previous_traj_id = traj_id
            previous_poi_id = poi_id

    return G


def save_graph_to_csv(G, dst_dir):
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    np.savetxt(os.path.join(dst_dir, 'graph_A.csv'), A.todense(), delimiter=',')

    with open(os.path.join(dst_dir, 'graph_X.csv'), 'w') as f:
        print('poi_id,checkin_cnt,latitude,longitude', file=f)
        for node_name, attr in G.nodes(data=True):
            print(f"{node_name},{attr['checkin_cnt']},"
                  f"{attr['latitude']},{attr['longitude']}", file=f)


def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, 'graph.pkl'), 'wb'))


def save_graph_edgelist(G, dst_dir):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join(dst_dir, 'graph_node_id2idx.txt'), 'w') as f:
        for node, idx in node_id2idx.items():
            print(f'{node}, {idx}', file=f)

    with open(os.path.join(dst_dir, 'graph_edge.edgelist'), 'w') as f:
        for u, v, data in G.edges(data=True):
            print(f"{node_id2idx[u]} {node_id2idx[v]} {data['weight']}", file=f)


def load_graph_adj_mtx(path):
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt',
                             feature3='latitude', feature4='longitude'):
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature3, feature4]]
    X = rlt_df.to_numpy()
    return X


def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    node_degrees = [deg for _, deg in G.degree()]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")


if __name__ == '__main__':
    dst_dir = r'../datasets/nyc/preprocessed/'

    train_df = pd.read_csv(os.path.join(dst_dir, 'train_sample.csv'))
    print('Build global POI checkin graph -----------------------------------')
    G = build_global_POI_checkin_graph(train_df)

    save_graph_to_pickle(G, dst_dir=dst_dir)
    save_graph_to_csv(G, dst_dir=dst_dir)
    save_graph_edgelist(G, dst_dir=dst_dir)
