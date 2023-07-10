def compute_permanence(G, community, neighbors, community_structure, nodes_compute_already, c_i):
    com_permanence = 0

    for i in range(len(community)):
        permanence = 0

        # degree of node i
        D_v = len(neighbors[community[i]])

        # clustering coefficient
        subgraph = G.induced_subgraph(community)

        vs_index = 0
        vs_name_temp = ""
        for vs_name in subgraph.vs["name"]:
            if vs_name == community[i]:
                vs_name_temp = vs_name
                break
            vs_index += 1

        c_in_v = subgraph.transitivity_local_undirected(vs_index, mode='zero')

        # internal degree
        I_v = len(set(neighbors[community[i]]).intersection(set(community)))

        # external connections
        E_max = 0
        for j in range(len(community_structure)):
            int_community_j = list(map(int, community_structure[j]))
            if j != c_i:
                E = len(set(neighbors[community[i]]).intersection(set(int_community_j)))
                if E > E_max:
                    E_max = E
        if E_max == 0:
            E_max = 1

        if vs_name_temp not in nodes_compute_already:
            nodes_compute_already.add(vs_name_temp)

            # print("------------------------")
            # print("I_v",I_v)
            # print("E_max", E_max)
            # print("D_v", D_v)
            # print("c_in_v", c_in_v)

            permanence = I_v / (E_max * D_v) + c_in_v - 1
            # print("result", permanence)
            com_permanence += permanence
            # print("com_permanence",com_permanence)
        # else:
        # print("already")
        """
        if E_max == 0:
            E_max = 1
        print("------------------------")
        print("I_v", I_v)
        print("E_max", E_max)
        print("D_v", D_v)
        print("c_in_v", c_in_v)

        permanence = I_v / (E_max * D_v) + c_in_v - 1
        com_permanence += permanence
        print("com_permanence",com_permanence)
        """
    # print("com_permanence" ,com_permanence)
    return com_permanence


def compute_coms_avg_permanence(G, neighbors, coms):
    coms_permanence = 0


    all_length = 0
    for c_i in range(len(coms)):
        # print("times")
        # print("com_single", com)
        # print("nodes_compute_already",nodes_compute_already)
        nodes_compute_already = set()
        coms_permanence += compute_permanence(G, coms[c_i], neighbors, coms, nodes_compute_already, c_i)
        # coms_permanence += compute_permanence(G, com, neighbors, coms)
        all_length += len(coms[c_i])
    return coms_permanence / all_length
