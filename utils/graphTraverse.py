import networkx as nx


def get_neighbors(edge_index, neighbor_order):
    """获取图的k阶邻居组成的节点。返回每个节点的neighbor_order的邻居相连的图。

    Args:
        edge_index ([type]): list[str, tgt]
        neighbor_order ([type]): int

    Returns:
        list: list(src, tgt)
    """ 
    G=nx.Graph()
    G.add_edges_from(edge_index)
    n_G = nx.Graph()
    new_edges=[]
    length = dict(nx.all_pairs_shortest_path_length(G))
    for node in list(G.nodes):
        n_G.add_edges_from([[node,t_node] for t_node in length[node] if length[node][t_node]==neighbor_order])
    return list(n_G.edges)



if __name__ == '__main__':
    
    G = nx.Graph()
    #G.add_edges_from([(1, 2),(2, 3),(4,5),(3, 4),(4, 1),(1, 5),(5, 6),(6, 7),(7, 8),(8, 9),(4, 10),(10, 11)])
    edge_index = [[2, 1],[2, 3],[4,5],[3, 4],[4, 1],[1, 5],[5, 6],[6, 7],[7, 8],[8, 9],[4, 10],[10, 11], [10, 11]]
    G.add_edges_from([[2, 1],[2, 3],[4,5],[3, 4],[4, 1],[1, 2],[3, 2],[5,4],[4, 3],[1, 4],[1, 5],[5, 6],[6, 7],[7, 8],[8, 9],[4, 10],[10, 11], [10, 11]])
    a = get_neighbors(edge_index, 3)
    print(a)

    #G = nx.path_graph(5)
    length = dict(nx.all_pairs_shortest_path_length(G))
    for node in [9,4,5,2]:
        print(f"1 - {node}: {length[1][node]}")
    print(list(G.edges))
    e = list(G.edges)
    import torch
    c = torch.tensor(e)
    print(c)
    print(list(G.nodes))