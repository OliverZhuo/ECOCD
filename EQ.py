def cal_Q(partition, G):

    m = len(G.edges(None, False))
    q = 0

    for community in partition:

        for i in range(len(community)):
            for j in range(len(community)):

                u_int = community[i]
                v_int = community[j]

                O_u = 0
                O_v = 0

                O_u_set = set()
                O_v_set = set()

                for c in partition:

                    #print("set u_int", O_u_set)
                    #print("set c", set(c))

                    O_u_set.add(u_int)
                    O_v_set.add(v_int)

                    if len(O_u_set.intersection(set(c))) != 0:
                        O_u += 1
                    if len(O_v_set.intersection(set(c))) != 0:
                        O_v += 1

                #print("**********")
                #print(O_u)
                #print(O_v)

                u = str(community[i])
                v = str(community[j])

                K_u = len([x for x in G.neighbors(str(u))])
                K_v = len([x for x in G.neighbors(str(v))])

                A_uv = 0.0

                if G.has_edge(u, v):
                    A_uv += 1.0

                q += (A_uv - (K_u * K_v) / (2 * m)) / (O_u * O_v)

    q /= (2*m)

    return q
