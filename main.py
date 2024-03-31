import copy
import math
import random
import numpy as np
import pandas as pd
from igraph import *
import networkx as nx


def load_community_membership(name_of_ground_truth):
    ground_truth_path = name_of_ground_truth
    lines = open(ground_truth_path, "r").readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
        lines[i] = lines[i].split(" ")

    num_max_community = 0
    for i in range(len(lines)):
        for j in range(1, len(lines[i])):
            if int(lines[i][j]) > num_max_community:
                num_max_community = int(lines[i][j])
    ground_truth_community_list = []

    for i in range(num_max_community + 1):
        ground_truth_community_list.append([])

    for i in range(len(lines)):
        for j in range(1, len(lines[i])):
            ground_truth_community_list[int(lines[i][j])].append(int(lines[i][0]))
    del (ground_truth_community_list[0])

    return ground_truth_community_list, num_max_community


def igraph_load_edge(dataset):
    path = dataset
    G = Graph().Read_Edgelist(path, directed=False)
    G.simplify(multiple=True)
    return G


def nndsvd_initialization(A, rank):
    u, s, v = np.linalg.svd(A, full_matrices=False)
    v = v.T
    w = np.zeros((A.shape[0], rank))
    h = np.zeros((rank, A.shape[1]))

    w[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = np.sqrt(s[0]) * np.abs(v[:, 0].T)

    for i in range(1, rank):
        ui = u[:, i]
        vi = v[:, i]
        ui_pos = (ui >= 0) * ui
        ui_neg = (ui < 0) * -ui
        vi_pos = (vi >= 0) * vi
        vi_neg = (vi < 0) * -vi

        ui_pos_norm = np.linalg.norm(ui_pos, 2)
        ui_neg_norm = np.linalg.norm(ui_neg, 2)
        vi_pos_norm = np.linalg.norm(vi_pos, 2)
        vi_neg_norm = np.linalg.norm(vi_neg, 2)

        norm_pos = ui_pos_norm * vi_pos_norm
        norm_neg = ui_neg_norm * vi_neg_norm

        if norm_pos >= norm_neg:
            w[:, i] = np.sqrt(s[i] * norm_pos) / ui_pos_norm * ui_pos
            h[i, :] = np.sqrt(s[i] * norm_pos) / vi_pos_norm * vi_pos.T
        else:
            w[:, i] = np.sqrt(s[i] * norm_neg) / ui_neg_norm * ui_neg
            h[i, :] = np.sqrt(s[i] * norm_neg) / vi_neg_norm * vi_neg.T
    return w, h


def random_initialization(A, rank):
    number_of_documents = A.shape[0]
    number_of_terms = A.shape[1]
    W = np.random.uniform(1, 2, (number_of_documents, rank))
    H = np.random.uniform(1, 2, (rank, number_of_terms))
    return W, H


def nmf(A, k, max_iter):
    # W, H = random_initialization(A, k)
    W, H = nndsvd_initialization(A, k)
    norms = []
    e = 1.0e-10
    for n in range(max_iter):
        print("n_iter", n)
        W_TA = W.T @ A
        W_TWH = W.T @ W @ H + e
        for i in range(np.size(H, 0)):
            for j in range(np.size(H, 1)):
                H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j]
        AH_T = A @ H.T
        WHH_T = W @ H @ H.T + e
        for i in range(np.size(W, 0)):
            for j in range(np.size(W, 1)):
                W[i, j] = W[i, j] * AH_T[i, j] / WHH_T[i, j]
        norm = np.linalg.norm(A - W @ H, 'fro')
        norms.append(norm)
    return W, H, norms


def EC():
    W, H, norms = nmf(A, k, 500)

    now_communities = list([] for i in range(k))
    i_max = [0 for j in range(len(W))]
    for i in range(len(W)):
        i_max_index = 0
        i_max[i] = 0
        for j in range(len(W[i])):
            if W[i][j] > i_max[i]:
                i_max[i] = W[i][j]
                i_max_index = j
        now_communities[i_max_index].append(i)

    core_communities_length = 0
    for i in range(len(now_communities)):
        core_communities_length += len(now_communities[i])
    core_communities = copy.deepcopy(now_communities)

    while True:
        now_communities = expansionWithContraction(G_igraph, neighbors, now_communities,
                                                   core_communities, W)




def expansionWithContraction(G_igraph, neighbors, now_communities, core_communities, W):
    random_C_id = random.randint(0, len(now_communities) - 1)
    candidate_num = math.ceil(len(now_communities[random_C_id]) * 0.1)
    add_num = math.ceil(len(now_communities[random_C_id]) / 10)

    minus_num = math.ceil(add_num / 3)

    if add_num == minus_num:
        minus_num = add_num - 1

    for i in range(add_num):
        all_neighbors = list()
        for node_id in now_communities[random_C_id]:
            all_neighbors.extend(neighbors[node_id])
        nodes_times = pd.value_counts(all_neighbors)

        candidate_nodes = list()
        for j in range(len(nodes_times)):
            if nodes_times.index[j] not in now_communities[random_C_id]:
                candidate_nodes.append(nodes_times.index[j])
            if len(candidate_nodes) >= candidate_num:
                break
        nodes_similarity = compute_similarity(candidate_nodes, neighbors, now_communities[random_C_id])
        nodes_distance = compute_distance(G_igraph, candidate_nodes, now_communities[random_C_id])
        nodes_s_d = list([0, 0.0] for i in range(len(candidate_nodes)))
        for i in range(len(candidate_nodes)):
            nodes_s_d[i][0] = nodes_similarity[i][0]
            nodes_s_d[i][1] = nodes_similarity[i][1] + nodes_distance[i][1] + W[nodes_s_d[i][0]][random_C_id]
        max_node_id = find_max_score(nodes_s_d)
        if max_node_id not in now_communities[random_C_id]:
            now_communities[random_C_id].append(max_node_id)

    for i in range(minus_num):
        nodes_permanence = compute_permanence(G_igraph, now_communities[random_C_id], neighbors,
                                              now_communities)
        min_node_id = find_min_permanence(nodes_permanence)
        if (min_node_id in now_communities[random_C_id]) and (min_node_id not in core_communities[random_C_id]):
            now_communities[random_C_id].remove(min_node_id)

    return now_communities


def find_min_permanence(nodes_permanence):
    min_permanence = 0
    min_node_id = 0
    for i in range(len(nodes_permanence)):
        if nodes_permanence[i][1] < min_permanence:
            min_node_id = nodes_permanence[i][0]
            min_permanence = nodes_permanence[i][1]
    return min_node_id


def find_max_score(nodes_similarity):
    max_permanence = 0
    max_node_id = 0
    for i in range(len(nodes_similarity)):
        if nodes_similarity[i][1] > max_permanence:
            max_node_id = nodes_similarity[i][0]
            max_permanence = nodes_similarity[i][1]
    return max_node_id


def compute_similarity(diff_nodes, neighbors, community):
    nodes_similarity = list([0, 0.0] for i in range(len(diff_nodes)))

    for i in range(len(diff_nodes)):
        inter_set = set(neighbors[diff_nodes[i]]).intersection(set(community))
        nodes_similarity[i][0] = diff_nodes[i]
        if len(community) != 0:
            nodes_similarity[i][1] = round(len(inter_set) / len(community), 4)
        else:
            print("division by zero")

    return nodes_similarity


def compute_distance(G_igraph, diff_nodes, community):
    nodes_distance = list([0, 0.0] for i in range(len(diff_nodes)))

    for i in range(len(diff_nodes)):
        shortest_paths = G_igraph.shortest_paths(source=diff_nodes[i], target=community, weights=None,
                                                 mode='all')
        sum_shortest_path_length = sum(shortest_paths[0])
        nodes_distance[i][0] = diff_nodes[i]

        if sum_shortest_path_length != 0:
            nodes_distance[i][1] = round(len(community) / sum_shortest_path_length, 4)

    return nodes_distance



def compute_permanence(G, community, neighbors, community_structure):
    nodes_permanence = list([0, 0.0] for i in range(len(community)))
    for i in range(len(community)):
        permanence = 0
        # degree of node i
        D_v = len(neighbors[community[i]])
        # clustering coefficient
        subgraph = G.induced_subgraph(community)
        vs_index = 0
        for vs_name in subgraph.vs["name"]:
            if vs_name == community[i]:
                break
            vs_index += 1
        c_in_v = subgraph.transitivity_local_undirected(vs_index, mode='zero')
        # internal degree
        I_v = len(set(neighbors[community[i]]).intersection(set(community))) - 1
        # external connections
        E_max = 0
        for j in range(len(community_structure)):
            int_community_j = list(map(int, community_structure[j]))
            if len(set(neighbors[community[i]]).intersection(set(int_community_j))) != 0:
                E_max += 1
        E_max -= 1

        if (E_max != 0) and (D_v != 0):
            permanence = I_v / (E_max * D_v) + c_in_v - 1

        nodes_permanence[i][0] = community[i]
        nodes_permanence[i][1] = round(permanence, 4)
    return nodes_permanence

folders = ["mu01", "mu02", "mu03", "mu04", "mu05", "om2", "om3", "om4", "om5", "om6", "on10", "on20", "on30",
           "on40", "on50", "n1000", "n2000", "n3000", "n4000", "n5000", "n6000", "n7000", "n8000", "n9000", "n10000"]

for folder in folders:

    path = "C:/Users/18859/Desktop/Now/LFR_Networks/" + folder + "/network.dat"
    ground_truth = "C:/Users/18859/Desktop/Now/LFR_Networks/" + folder + "/community.dat"
    ground_truth_community, max_community = load_community_membership(ground_truth)
    k = max_community

    G_nx = nx.read_edgelist(path, create_using=nx.Graph())
    G_nx.add_node("0")
    G_igraph = igraph_load_edge(path)
    A = G_igraph.get_adjacency()
    A = np.array(A.data)

    vertices = []
    max_id = 0
    for idx in enumerate(G_igraph.vs):
        if idx[0] > max_id:
            max_id = idx[0]
        vertices.append(idx[0])
    neighbors = G_igraph.neighborhood()
    G_igraph.vs["name"] = [i for i in range(max_id + 1)]

    EC()
