"""
 Name:         ling_model
 Author:       小志
 Date:         2020/10/31
"""
import multinetx as mx
import numpy as np
import random


def create_MultilayerGraph(N, p_g1=0.4, p_g2=0.4, seed_g1=None, seed_g2=None, intra_layer_edges_weight=None, inter_layer_edges_weight=None):
    """Create a two-layer complex network instance object and return
    创建双层复杂网络实例对象并返回
    Parameters
    ----------
    N : int
        Number of nodes per layer.
    p_g1 : float
           Probability for edge creation of the lower layer (default=0.4).
    p_g2 : float
           Probability for edge creation of the upper layer (default=0.4).
    seed_g1 : int, optional
              Seed of the lower random number generator (default=None).
    seed_g2 : int, optional
              Seed of the upper random number generator (default=None).
    intra_layer_edges_weight |  inter_layer_edges_weight :
            Set the weights of the MultilayerGraph edges
            Set the "intra_layer_edges_weight" and "inter_layer_edges_weight"
            as an edge attribute with the name "weight"
    Return :  a MultilayerGraph object.
    """
    # 创建随机网络
    g1 = mx.generators.erdos_renyi_graph(n=N, p=p_g1, seed=seed_g1)
    g2 = mx.generators.erdos_renyi_graph(n=N, p=p_g2, seed=seed_g2)

    # 创建所有随机网络对应的超矩阵
    adj_block = mx.lil_matrix(np.zeros((N * 2, N * 2)))
    adj_block[0:  N, N:2 * N] = np.identity(N)
    adj_block += adj_block.T

    # 生成多层复杂网络对象实例
    mg = mx.MultilayerGraph(list_of_layers=[g1, g2], inter_adjacency_matrix=adj_block)
    # 设置层内以及层间边缘权重
    mg.set_edges_weights(intra_layer_edges_weight=intra_layer_edges_weight,
                         inter_layer_edges_weight=inter_layer_edges_weight)
    return mg


def random_0_model(Graph, N, nswap=1, max_tries=100, connected=True):
    """Zero-order zero model based on broken edge reconnection and return
    基于断边重连的0阶零模型

    Parameters
    ----------
    Graph : object
            a MultilayerGraph object.
    N : int
        Number of nodes per layer (N >= 4).
    nswap : int
           Number of successful reconnections (default=1).
    max_tries : int
                Maximum number of reconnection attempts (default=100).
    connected : boolean
                Check network connectivity (default=True[Keep connected]).

    Return :  A zero-order two-layer complex network zero model.
    """
    assert mx.is_directed(Graph) is False, \
        "It is only allowed for undirected complex network"
    assert nswap < max_tries, \
        "Number of swaps > number of tries allowed."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."

    G = Graph
    n = 0  # 已尝试重连进行的次数
    swapcount = 0  # 已重连成功的次数
    edges_up = G.get_intra_layer_edges_of_layer(layer=1)
    nodes_up = list(G.nodes)[N:2 * N]
    while swapcount < nswap:
        n = n + 1
        u, v = random.choice(edges_up)  # 随机的选最上层网络中的一条要断开的边
        x, y = random.sample(nodes_up, 2)  # 随机从最上层中找两个不相连的节点
        if len(set([u, v, x, y])) < 4:
            continue
        if (x, y) not in edges_up and (y, x) not in edges_up:
            G.remove_edge(u, v)  # 断旧边
            G.add_edge(x, y)  # 连新边
            edges_up.remove((u, v))
            edges_up.append((x, y))

            if connected == True:  # 判断是否需要保持联通特性
                if not mx.is_connected(G):  # 保证网络是全联通的:若网络不是全联通网络，则撤回交换边的操作
                    print(mx.is_connected(G))
                    G.add_edge(u, v)
                    G.remove_edge(x, y)
                    edges_up.remove((x, y))
                    edges_up.append((u, v))
                    continue
            swapcount = swapcount + 1
        if n >= max_tries:
            print(f"Maximum number of swap attempts {n} exceeded before desired swaps achieved {nswap}.")
            break
    return G

def write_data_toDat(fname, Graph, N, weight=None):
    """Write hypermatrix data to text file
    写入超矩阵数据到文本文件中

    Parameters
    ----------
    fname : filename or file handle
        If the filename ends in ``.gz``, the file is automatically saved in
        compressed gzip format.  `loadtxt` understands gzipped files
        transparently.
    Graph : object
            a MultilayerGraph object.
    weight : string or None, optional (default='weight')
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.
    """
    matrix_data = mx.adjacency_matrix(Graph,weight=weight).todense()
    assert int(matrix_data.shape[0]/2) == N,\
    "传入的矩阵中单层节点个数要与传入的N节点相等"
    frontff_dat = f"""DL\nN={N} NM=2\nFORMAT = FULLMATRIX DIAGONAL PRESENT\nDATA:\n"""
    arr_up = matrix_data.getA()[N:2*N,N:2*N]
    arr_down = matrix_data.getA()[0:N,0:N]
    arr_data = np.vstack((arr_up, arr_down))
    np.savetxt(fname, arr_data, fmt='%s', delimiter=' ')
    file_name = fname
    with open(file_name, 'r') as f:
        lines = f.readlines()
    with open(file_name, 'w') as n:
        n.write(frontff_dat)
        n.writelines(lines)
