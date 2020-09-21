import numpy as np
import scipy.sparse as sp
import torch
import json
import sys
import os
import pickle as pkl
import networkx as nx
from networkx.readwrite import json_graph
from time import perf_counter

NoselfNone = 0
NoselfSym = 1
NoselfAsym = 2
SelfNone = 3
SelfSym = 4
SelfAsym = 5

def parse_index_file(filename):
    """Parse index file."""
    index = []
    fo = open(filename)
    for line in fo:
        index.append(int(line.strip()))
    fo.close()
    return index

def aug_normalized_adjacency(adj, how_agg=SelfSym):
    if how_agg > 2:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    if how_agg % 3 == 0:
        return adj.tocoo()
    elif how_agg % 3 == 1:
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    elif how_agg % 3 == 2:
        row_inv = np.power(row_sum, -1.0).flatten()
        row_inv[np.isinf(row_inv)] = 0.
        row_mat_inv = sp.diags(row_inv)
        return adj.dot(row_mat_inv).transpose().tocoo()

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def to_list(graph):
    return nx.to_edgelist(nx.from_dict_of_lists(graph))

def to_torch_sparse_tensor(graph, how_agg=SelfSym):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(graph)
    sparse_mx = aug_normalized_adjacency(adj, how_agg)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str="cora"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    graph = nx.from_dict_of_lists(graph)
    test_idx_reorder = parse_index_file("data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # SGNN
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    # self-loop
    features = row_normalize(features)

    # sample neighbors
    id2idx = {id: id for id in graph.nodes()}
    graph, _ = construct_adj(graph, id2idx)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels_raw = torch.FloatTensor(labels)
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    label_max = torch.max(labels) - torch.min(labels) + 1
    label_max = label_max.item()

    idx = {}
    idx["train"] = torch.LongTensor(idx_train)
    idx["val"] = torch.LongTensor(idx_val)
    idx["test"] = torch.LongTensor(idx_test)

    return graph, features, labels, label_max, idx, labels_raw

def construct_adj(G, id2idx, inductive=False):
    adj = {}
    for i in range(len(id2idx)):
        adj[i] = []
    deg = np.zeros((len(id2idx),))

    for nodeid in G.nodes():
        if inductive and (G.node[nodeid]['test'] or G.node[nodeid]['val']):
            continue
        neighbors = np.array([id2idx[neighbor]
            for neighbor in G.neighbors(nodeid)
            if not inductive or (not G[nodeid][neighbor]['train_removed'])])
        deg[id2idx[nodeid]] = len(neighbors)
        if len(neighbors) == 0:
            continue
        adj[id2idx[nodeid]] = neighbors
    return adj, deg

def construct_test_adj(G, id2idx):
    adj = {}
    for i in range(len(id2idx)):
        adj[i] = []
    for nodeid in G.nodes():
        neighbors = np.array([id2idx[neighbor]
            for neighbor in G.neighbors(nodeid)])
        if len(neighbors) == 0:
            continue
        adj[id2idx[nodeid]] = neighbors
    return adj

def load_inductive_dataset(dataset_str="ppi", normalize=True, load_walks=False):
    """
    Load Inductive Networks Datasets.
    """
    prefix = "data/{}/{}".format(dataset_str.lower(), dataset_str.lower())
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    temp_labels = list(class_map.values())[0]
    if isinstance(temp_labels, list):
        lab_conversion = lambda n : np.array(n)
        labels = np.zeros((len(id_map), len(temp_labels)))
    else:
        lab_conversion = lambda n : int(n)
        labels = np.zeros((len(id_map)))

    for k,v in class_map.items():
        labels[id_map[conversion(k)]] = lab_conversion(v)

    if isinstance(list(class_map.values())[0], list):
        label_max = len(labels[0])
        labels = torch.FloatTensor(labels)
    else:
        label_max = np.max(labels) - np.min(labels) + 1
        label_max = label_max.astype(np.int64)
        labels = torch.LongTensor(labels)

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    graph, deg = construct_adj(G, id_map, inductive=True)
    graph_test = construct_test_adj(G, id_map)

    idx_val = [id_map[n] for n in G.nodes() if G.node[n]['val']]
    idx_test = [id_map[n] for n in G.nodes() if G.node[n]['test']]

    no_train_nodes_set = set(idx_val + idx_test)
    idx_train = set(id_map.values()).difference(no_train_nodes_set)
    # don't train on nodes that only have edges to test set
    idx_train = [n for n in idx_train if deg[n] > 0]

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_feats = feats[idx_train]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    # porting to pytorch
    feats = torch.FloatTensor(feats).float()
    idx = {}
    idx["train"] = torch.LongTensor(idx_train)
    idx["val"] = torch.LongTensor(idx_val)
    idx["test"] = torch.LongTensor(idx_test)

    return graph, graph_test, feats, labels, label_max, idx

