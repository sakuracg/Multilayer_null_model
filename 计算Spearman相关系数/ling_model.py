"""
 Name:         ling_model
 Author:       小志
 Date:         2020/10/31
"""
import multinetx as mx
import numpy as np
import random

def create_MultilayerGraph(N, p_g1=0.1, p_g2=0.1, seed_g1=None, seed_g2=None, intra_layer_edges_weight=None, inter_layer_edges_weight=None):
    """Create a two-layer complex network instance object and return
    创建双层复杂网络实例对象并返回
    Parameters
    ----------
    N : int
        Number of nodes per layer.
    p_g1 : float
           Probability for edge creation of the lower layer (default=0.1).
    p_g2 : float
           Probability for edge creation of the upper layer (default=0.1).
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

    n = 0  # 已尝试重连进行的次数
    swapcount = 0  # 已重连成功的次数
    edges_up = Graph.get_intra_layer_edges_of_layer(layer=0)
    nodes_up = list(Graph.nodes)[0:N]
    while swapcount < nswap:
        n = n + 1
        u, v = random.choice(edges_up)  # 随机的选最上层网络中的一条要断开的边
        x, y = random.sample(nodes_up, 2)  # 随机从最上层中找两个不相连的节点
        if len(set([u, v, x, y])) < 4:
            continue
        if (x, y) not in edges_up and (y, x) not in edges_up:
            Graph.remove_edge(u, v)  # 断旧边
            Graph.add_edge(x, y)  # 连新边
            edges_up.remove((u, v))
            edges_up.append((x, y))

            if connected == True:  # 判断是否需要保持联通特性
                if not mx.is_connected(Graph):  # 保证网络是全联通的:若网络不是全联通网络，则撤回交换边的操作                    
                    Graph.add_edge(u, v)
                    Graph.remove_edge(x, y)
                    edges_up.remove((x, y))
                    edges_up.append((u, v))
                    continue
            swapcount = swapcount + 1
        if n >= max_tries:
            print(f"Maximum number of swap attempts {n} exceeded before desired swaps achieved {nswap}.")
            break
    return Graph

def random_1_model(Graph, N, nswap=1, max_tries=100, connected=True, analy_layer=0):
    """Zero-order level model based on broken edge reconnection and return
        基于断边重连的1阶零模型  PS：保持度分布不变

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
        analy_layer : int (0 | 1)
                    choose analy layer.

        Return :  A level-order two-layer complex network zero model.
        """
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert nswap <= max_tries, \
        "Number of swaps > number of tries allowed."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."

    n = 0  # 已尝试重连进行的次数
    swapcount = 0  # 已重连成功的次数

    # 求得双层网络[所有节点]的[从0到结束]的有序的键值对元组
    tem_keys, tem_degrees = zip(*Graph.degree())  # zip的过程，zip(*)可把元组反解压
    # 对[要分析的层的节点（取出来）]的有序键值对元组进行随机化
    keys, degrees = random_nodes_degrees(tem_keys[analy_layer*N:(analy_layer+1)*N], tem_degrees[analy_layer*N:(analy_layer+1)*N])
    cdf = mx.utils.cumulative_distribution(degrees)  # 从离散分布返回归一化的累积分布。

    while swapcount < nswap:
        if n >= max_tries:
            print(f'尝试次数 {n} 已超过允许的最大尝试次数，而交换成功的次数为{swapcount}')
            break

        n += 1
        # 在保证度分布不变的情况下，随机选取两条连边u-v，x-y
        (ui, xi) = mx.utils.discrete_sequence(2, cdistribution=cdf)
        if ui == xi:
            continue
        u = keys[ui]  # 已经随机化的了
        x = keys[xi]

        # 求已经选取的两个节点u，x，他们两个的相连节点v，y

        # 得到第0层（默认分析第0层，变量名也是这样）的边
        edges_0 = get_edge_0(Graph=Graph, analy_layer=analy_layer)
        # 得到u，x所有边的相连节点列表集合
        u_connect_nodes = get_connect_nodes(u, edges_0)
        x_connect_nodes = get_connect_nodes(x, edges_0)
        if len(x_connect_nodes) == 0 or len(u_connect_nodes) == 0:
            continue
        v = random.choice(u_connect_nodes)
        y = random.choice(x_connect_nodes)

        if len(set([u, v, x, y])) == 4:  # 保证是四个独立节点
            if (y not in Graph[u]) and (v not in Graph[x]):  # 保证新生成的连边是原网络中不存在的边
                Graph.add_edge(u,y)                           #增加两条新连边
                Graph.add_edge(v,x)
                Graph.remove_edge(u,v)
                Graph.remove_edge(x,y)
               
                if connected == True:  # 判断是否需要保持联通特性，为1的话则需要保持该特性
                    if not mx.is_connected(Graph):  # 保证网络是全联通的:若网络不是全联通网络，则撤回交换边的操作
                        Graph.add_edge(u,v)
                        Graph.add_edge(x,y)
                        Graph.remove_edge(u,y)
                        Graph.remove_edge(x,v)
                        continue
                swapcount = swapcount + 1
    return Graph


def random_2_model(Graph, N, nswap=1, max_tries=100, connected=True, analy_layer=0):
    """Zero-order two model based on broken edge reconnection and return
        基于断边重连的2阶零模型  PS：保持度分布不变   以及两条边连接节点的度值不变

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
        analy_layer : int (0 | 1)
                    choose analy layer.

        Return :  A two-order two-layer complex network zero model.
        """
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert nswap <= max_tries, \
        "Number of swaps > number of tries allowed."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."

    n = 0  # 已尝试重连的次数
    swapcount = 0  # 已尝试重连成功的次数

    # 求得双层网络[所有节点]的[从0到结束]的有序的键值对元组
    tem_keys, tem_degrees = zip(*Graph.degree())
    # 对[要分析的层的节点（取出来）]的有序键值对元组进行随机化
    keys, degrees = random_nodes_degrees(tem_keys[analy_layer * N:(analy_layer + 1) * N],
                                         tem_degrees[analy_layer * N:(analy_layer + 1) * N])
    # 求cdf
    cdf = mx.utils.cumulative_distribution(degrees)

    while swapcount < nswap:
        if n >= max_tries:
            print(f'尝试次数 {n} 已超过允许的最大尝试次数，而交换成功的次数为{swapcount}')
            break

        n += 1
        # 保持度分布不变的情况下 随机选取两条边
        # 疑问： 根据cdf 求得的ui，xi是什么关系？？？？
        (ui, xi) = mx.utils.discrete_sequence(2, cdistribution=cdf)
        if ui == xi:
            continue
        u = keys[ui]  # 已经随机化的了
        x = keys[xi]

        # 求已经选取的两个节点u，x，他们两个的相连节点v，y

        # 得到第0层（默认分析第0层，变量名也是这样）的边
        edges_0 = get_edge_0(Graph=Graph, analy_layer=analy_layer)
        # 得到u，x所有边的相连节点列表集合
        u_connect_nodes = get_connect_nodes(u, edges_0)
        x_connect_nodes = get_connect_nodes(x, edges_0)
        if len(x_connect_nodes) == 0 or len(u_connect_nodes) == 0:
            continue
        # 随机从集合中选取一个点
        v = random.choice(u_connect_nodes)
        y = random.choice(x_connect_nodes)

        if len(set([u, v, x, y])) == 4:
            if Graph.degree(v) == Graph.degree(y):  # 保证节点的度匹配特性不变
                if (y not in Graph[u]) and (v not in Graph[x]):  # 保证新生成的连边是原网络中不存在的边
                    Graph.add_edge(u, y)  # 增加两条新连边
                    Graph.add_edge(v, x)
                    Graph.remove_edge(u, v)
                    Graph.remove_edge(x, y)

                    if connected == True:  # 判断是否需要保持联通特性，为1的话则需要保持该特性
                        if not mx.is_connected(Graph):  # 保证网络是全联通的:若网络不是全联通网络，则撤回交换边的操作
                            Graph.add_edge(u, v)
                            Graph.add_edge(x, y)
                            Graph.remove_edge(u, y)
                            Graph.remove_edge(x, v)
                            continue
                    swapcount = swapcount + 1
    return Graph


def random_25_model(Graph, N, nswap=1, max_tries=100, connected=True, analy_layer=0):
    """Zero-order two point five model based on broken edge reconnection and return
        基于断边重连的2.5零模型  PS：保持度分布不变   以及两条边连接节点的度值不变

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
        analy_layer : int (0 | 1)
                    choose analy layer.

        Return :  A two point five-order two-layer complex network zero model.
        """

    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert nswap <= max_tries, \
        "Number of swaps > number of tries allowed."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."

    # 由于multiNetx库复制网络时会遇到一些问题（比如复制的网络不能取到边），但结构是可以用的。所以把复制的网络当做原始网络，把原网络当做要分析的网络
    tem_Graph = mx.Graph.copy(Graph)
    n = 0  # 已尝试重连的次数
    swapcount = 0  # 已尝试重连成功的次数

    # 求得双层网络[所有节点]的[从0到结束]的有序的键值对元组
    tem_keys, tem_degrees = zip(*Graph.degree())
    # 对[要分析的层的节点（取出来）]的有序键值对元组进行随机化
    keys, degrees = random_nodes_degrees(tem_keys[analy_layer * N:(analy_layer + 1) * N],
                                         tem_degrees[analy_layer * N:(analy_layer + 1) * N])
    # 求cdf
    cdf = mx.utils.cumulative_distribution(degrees)

    while swapcount < nswap:
        if n >= max_tries:
            print(f'尝试次数 {n} 已超过允许的最大尝试次数，而交换成功的次数为{swapcount}')
            break

        n += 1
        # 保持度分布不变的情况下 随机选取两条边
        (ui, xi) = mx.utils.discrete_sequence(2, cdistribution=cdf)
        if ui == xi:
            continue
        u = keys[ui]  # 已经随机化的了
        x = keys[xi]

        # 求已经选取的两个节点u，x，他们两个的相连节点v，y

        # 得到第0层（默认分析第0层，变量名也是这样）的边
        edges_0 = get_edge_0(Graph=Graph, analy_layer=analy_layer)
        # 得到u，x所有边的相连节点列表集合
        u_connect_nodes = get_connect_nodes(u, edges_0)
        x_connect_nodes = get_connect_nodes(x, edges_0)
        if len(x_connect_nodes) == 0 or len(u_connect_nodes) == 0:
            continue
        # 随机从集合中选取一个点
        v = random.choice(u_connect_nodes)
        y = random.choice(x_connect_nodes)

        if len(set([u, v, x, y])) == 4:
            if Graph.degree(v) == Graph.degree(y):  # 保证节点的度匹配特性不变
                if (y not in Graph[u]) and (v not in Graph[x]):  # 保证新生成的连边是原网络中不存在的边
                    Graph.add_edge(u, y)  # 增加两条新连边
                    Graph.add_edge(v, x)
                    Graph.remove_edge(u, v)
                    Graph.remove_edge(x, y)

                    # 取到
                    degree_node_list = map(lambda t: (t[1], t[0]), tem_Graph.degree(
                        [u, v, x, y] + list(Graph[u]) + list(Graph[v]) + list(Graph[x]) + list(Graph[y])))

                    # 先找到四个节点以及他们邻居节点的集合，然后取出这些节点所有的度值对应的节点，格式为（度，节点）形式的列表
                    D = dict_degree_nodes(degree_node_list)  # 找到每个度对应的所有节点，具体形式为

                    j = 0
                    for i in range(len(D)):
                        avctem_Graph = mx.average_clustering(tem_Graph, nodes=set(list(D.values())[i]), weight=None,
                                                             count_zeros=True)
                        avcGraph = mx.average_clustering(Graph, nodes=set(list(D.values())[i]), weight=None,
                                                         count_zeros=True)
                        j += 1
                        if avctem_Graph != avcGraph:  # 若置乱前后度相关的聚类系数不同，则撤销此次置乱操作
                            Graph.add_edge(u, v)
                            Graph.add_edge(x, y)
                            Graph.remove_edge(u, y)
                            Graph.remove_edge(x, v)
                            break

                    if j != len(D):  # 度相关的聚类系数没有全部通过则跳出
                        continue

                    if connected == True:  # 判断是否需要保持联通特性，为1的话则需要保持该特性
                        if not mx.is_connected(Graph):  # 保证网络是全联通的:若网络不是全联通网络，则撤回交换边的操作
                            Graph.add_edge(u, v)
                            Graph.add_edge(x, y)
                            Graph.remove_edge(u, y)
                            Graph.remove_edge(x, v)
                            continue
                    swapcount = swapcount + 1
    return Graph


def random_3_model(Graph, N, nswap=1, max_tries=100, connected=True, analy_layer=0):
    """Zero-order three model based on broken edge reconnection and return
        基于断边重连的3零模型  PS：保持度分布不变   以及两条边连接节点的度值不变

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
        analy_layer : int (0 | 1)
                    choose analy layer.

        Return :  A three-order two-layer complex network zero model.
        """
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert nswap <= max_tries, \
        "Number of swaps > number of tries allowed."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."

    tem_Graph = mx.Graph.copy(Graph)
    n = 0  # 已尝试重连的次数
    swapcount = 0  # 已尝试重连成功的次数

    # 求得双层网络[所有节点]的[从0到结束]的有序的键值对元组
    tem_keys, tem_degrees = zip(*Graph.degree())
    # 对[要分析的层的节点（取出来）]的有序键值对元组进行随机化
    keys, degrees = random_nodes_degrees(tem_keys[analy_layer * N:(analy_layer + 1) * N],
                                         tem_degrees[analy_layer * N:(analy_layer + 1) * N])
    # 求cdf
    cdf = mx.utils.cumulative_distribution(degrees)

    while swapcount < nswap:
        if n >= max_tries:
            print(f'尝试次数 {n} 已超过允许的最大尝试次数，而交换成功的次数为{swapcount}')
            break

        n += 1
        # 保持度分布不变的情况下 随机选取两条边
        (ui, xi) = mx.utils.discrete_sequence(2, cdistribution=cdf)
        if ui == xi:
            continue
        u = keys[ui]  # 已经随机化的了
        x = keys[xi]

        # 求已经选取的两个节点u，x，他们两个的相连节点v，y

        # 得到第0层（默认分析第0层，变量名也是这样）的边
        edges_0 = get_edge_0(Graph=Graph, analy_layer=analy_layer)
        # 得到u，x所有边的相连节点列表集合
        u_connect_nodes = get_connect_nodes(u, edges_0)
        x_connect_nodes = get_connect_nodes(x, edges_0)
        if len(x_connect_nodes) == 0 or len(u_connect_nodes) == 0:
            continue
        # 随机从集合中选取一个点
        v = random.choice(u_connect_nodes)
        y = random.choice(x_connect_nodes)

        if len(set([u, v, x, y])) == 4:
            if Graph.degree(v) == Graph.degree(y):  # 保证节点的度匹配特性不变
                if (y not in Graph[u]) and (v not in Graph[x]):  # 保证新生成的连边是原网络中不存在的边
                    Graph.add_edge(u, y)  # 增加两条新连边
                    Graph.add_edge(v, x)
                    Graph.remove_edge(u, v)
                    Graph.remove_edge(x, y)

                    node_list = [u, v, x, y] + list(Graph[u]) + list(Graph[v]) + list(Graph[x]) + list(
                        Graph[y])  # 找到四个节点以及他们邻居节点的集合
                    avctem_Graph = mx.clustering(tem_Graph, nodes=node_list)  # 计算旧网络中4个节点以及他们邻居节点的聚类系数
                    avcGraph = mx.clustering(Graph, nodes=node_list)  # 计算新网络中4个节点以及他们邻居节点的聚类系数

                    if avctem_Graph != avcGraph:  # 若置乱前后度相关的聚类系数不同，则撤销此次置乱操作
                        Graph.add_edge(u, v)
                        Graph.add_edge(x, y)
                        Graph.remove_edge(u, y)
                        Graph.remove_edge(x, v)
                        continue

                    if connected == True:  # 判断是否需要保持联通特性，为1的话则需要保持该特性
                        if not mx.is_connected(Graph):  # 保证网络是全联通的:若网络不是全联通网络，则撤回交换边的操作
                            Graph.add_edge(u, v)
                            Graph.add_edge(x, y)
                            Graph.remove_edge(u, y)
                            Graph.remove_edge(x, v)
                            continue
                    swapcount = swapcount + 1
    return Graph

def random_changenode_model(Graph, N, nswap=1, analy_layer=0):
    """Zero-order three model based on broken edge reconnection and return
        基于节点置乱零模型

        Parameters
        ----------
        Graph : object
                a MultilayerGraph object.
        N : int
            Number of nodes per layer (N >= 4).
        nswap : int
               Number of successful reconnections (default=1).
        analy_layer : int (0 | 1)
                    choose analy layer.

        Return :  A Node scrambling omplex network zero model.
        """

    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."

    swapcount = 0  # 已尝试交换节点并成功的次数

    while swapcount < nswap:
        # 随机取两个节点
        nodes = list(Graph.nodes)[analy_layer*N : (analy_layer+1)*N]
        x_node = random.choice(nodes)
        y_node = random.choice(nodes)
        print(x_node, y_node)
        if x_node == y_node:
            continue
        # 两个节点进行交换
        swap_one_node(Graph, x_node, y_node)
        # 加一
        swapcount += 1

    return Graph

def matchEffect_0_model(Graph, N, nswap=1, pattern='positive', analy_layer=0):
    """Zero model network with positive or negative matching effect and return
        具有正匹配或负匹配效应的零模型网络

        Parameters
        ----------
        Graph : object
                a MultilayerGraph object.
        N : int
            Number of nodes per layer (N >= 4).
        nswap : int
               Number of successful reconnections (default=1).
        pattern : str
                positive | negative
                Positive matching effect | Negative matching effect
        analy_layer : int (0 | 1)
                    choose analy layer.

        Return :  A complex network zero model.
        """
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."

    # 深度copy原始网络，为了保留原始网络的结构
    tem_Graph = mx.Graph.copy(Graph)
    swapcount = 0  # 已尝试交换节点并成功的次数

    while swapcount < nswap:
        # 随机取两个节点
        nodes = list(tem_Graph.nodes)[:N]
        x_node = random.choice(nodes)
        y_node = random.choice(nodes)

        if tem_Graph.degree(x_node) == tem_Graph.degree(y_node):
            continue

        # 得到度大节点和度小节点
        max_degree_node = max((tem_Graph.degree(x_node), x_node), (tem_Graph.degree(y_node), y_node))
        min_degree_node = min((tem_Graph.degree(x_node), x_node), (tem_Graph.degree(y_node), y_node))

        # 判断下层，是否也是这样，如果是，则跳出，继续下一次，如果不是，则交换
        # 先判断是正匹配还是负匹配模式
        if pattern == 'positive':
            if tem_Graph.degree(max_degree_node[1] + N) >= tem_Graph.degree(min_degree_node[1] + N):
                continue
        elif pattern == 'negative':
            if tem_Graph.degree(max_degree_node[1] + N) <= tem_Graph.degree(min_degree_node[1] + N):
                continue
        # 交换
        #这里为什么还是原始网络，因为深度复制的网络会报错，具体报错细节请自行运行好吧，
        swap_one_node(Graph, x_node, y_node, analy_layer=0)
        swapcount += 1

    return tem_Graph

def swap_one_node(Graph, node_1, node_2, analy_layer=0):
    """
    把 analy_layer 这个层中的两个节点进行置换，保持网络结构不变
    :param Graph: object
                a MultilayerGraph object.
    :param node_1: node 1 （int）
    :param node_2: node 2 （int）
    :param analy_layer: int (0 | 1)
                    choose analy layer.

    :return: None
    """
    swapped_edges = []
    edges = get_edge_0(Graph=Graph, analy_layer=analy_layer)
    for item in edges:
        if item.count(node_1) == 0 and item.count(node_2) == 0:
            continue
        elif item.count(node_1) == 1 and item.count(node_2) == 1:
            continue
        elif item.count(node_1) == 1 and item.count(node_2) == 0:
            v, w = item
            if (v,w) not in swapped_edges and (w,v) not in swapped_edges:
                Graph.remove_edge(v, w)
            tem_n = w if v == node_1 else v
            Graph.add_edge(node_2, tem_n)
            swapped_edges.append((node_2,tem_n))
        elif item.count(node_1) == 0 and item.count(node_2) == 1:
            v, w = item
            if (v,w) not in swapped_edges and (w,v) not in swapped_edges:
                Graph.remove_edge(v, w)
            tem_n = w if v == node_2 else v
            Graph.add_edge(node_1, tem_n)
            swapped_edges.append((node_1,tem_n))

def dict_degree_nodes(degree_node_list):
    """返回的字典为{度：[节点1，节点2，..], ...}，其中节点1和节点2有相同的度"""
    D = {}
    for degree_node_i in degree_node_list:
        if degree_node_i[0] not in D:
            D[degree_node_i[0]] = [degree_node_i[1]]
        else:
            D[degree_node_i[0]].append(degree_node_i[1])
    return D

def sub_edgesBigAndSmall(edges_list1, edges_list2):
    """两个边列表做差集"""
    sub_edges = []
    edges_all = edges_list1 + edges_list2
    for x,y in edges_all:
        if ((x,y) in edges_list1 or (y,x) in edges_list1) and ((x,y) in edges_list2 or (y,x) in edges_list2):
            continue
        else:
            sub_edges.append((x,y))
    return sub_edges

def get_edge_0(Graph, analy_layer):
    """1阶零模型算法中求每次断边重连时第0层的边（时时变化的只有get_intra_layer_edges()总边）

        analy_layer : int (0 | 1)
            choose analy layer.
        """
    if Graph.get_intra_layer_edges() == []:
        return '该网络非原始网络，而是经过Copy函数复制的函数，请修改为原始网络对象'
    intra_edges = list(Graph.get_intra_layer_edges())
    inter_edges = list(Graph.get_inter_layer_edges())
    sub_edges = sub_edgesBigAndSmall(intra_edges, inter_edges)

    intra_edges_layer_one = list(Graph.get_intra_layer_edges_of_layer(layer=1 if analy_layer == 0 else 0))
    sub_edges_one = sub_edgesBigAndSmall(sub_edges, intra_edges_layer_one)
    return sub_edges_one


def get_connect_nodes(node, edges):
    """1阶零模型算法中求一个节点集所相连边的另一个节点集"""
    connect_nodes = []
    for edge in edges:
        if node in edge:
            connect_nodes.append([val for val in edge if val != node][0])
    return connect_nodes

def random_nodes_degrees(keys, degrees):
    """Randomize the node and the corresponding degree one-to-one and return.
        将节点与对应的度进行一一对应的随机化并返回

        keys: tuple or list
            nodes
        degrees: tuple or list
            Node degrees
        Return: Nodes and degrees after randomization.
    """
    # 元组转换为可变的列表
    tem_keys = list(keys)

    # 随机化
    random.shuffle(tem_keys)

    # 根据tem_keys取到随机化的度
    tem_degrees = [degrees[i] for i in tem_keys]

    return tuple(tem_keys), tuple(tem_degrees)


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
    arr_data = np.vstack((arr_down, arr_up))
    np.savetxt(fname, arr_data, fmt='%s', delimiter=' ')
    file_name = fname
    with open(file_name, 'r') as f:
        lines = f.readlines()
    with open(file_name, 'w') as n:
        n.write(frontff_dat)
        n.writelines(lines)


# 计算相关系数
def get_degree_by_node(Graph, node, N, layer=0):
    """ 计算layer层node节点的度，注意这里进行了减一，减去了层间连线带来的度

    :param Graph: object |
            A MultilayerGraph object.
    :param node:  int |
            A node
    :param N: int |
            Number of nodes per layer
    :param layer: int |
            Number of layers
    :return: Degree of the node in the layer
    """
    return Graph.degree(N*layer + node) - 1

def get_degrees_by_layer(Graph, N, layer):
    """ 计算layer层所有节点的度，并以一个数组返回

    :param Graph: object |
            A MultilayerGraph object.
    :param N: int |
            Number of nodes per layer
    :param layer: int |
            Number of layers
    :return: A list of degrees of all nodes in layer
    """
    degrees = []
    for i in range(N):
        degrees.append(get_degree_by_node(Graph, i, N, layer))
    return degrees

def get_avg_degree_product(array1, array2, N):
    """ 计算出两层网络度乘积的平均数

    :param array1: numpy.ndarray |
            A numpy array
    :param array2: numpy.ndarray |
            A numpy array
    :param N: int |
            Number of nodes per layer
    :return: The average of the product of the two-layer network degree
    """
    return np.sum(array1*array2) / N

def get_avg_degree(array, N):
    """ 计算一层网络的平均度

    :param array: numpy.ndarray |
            A numpy array
    :param N: int |
            Number of nodes per layer
    :return: Average degree of the network
    """
    return np.sum(array) / N

def get_deominator_one(array, N):
    """ 计算分母中其中一个，我实在不知道那个玩意叫啥

    :param array: numpy.ndarray |
            A numpy array
    :param N: int |
            Number of nodes per layer
    :return:
    """
    a1 = get_avg_degree_product(array, array, N)
    a2 = np.square(get_avg_degree(array, N))
    return np.sqrt(a1 - a2)

def pearson_correlation_coefficient(Graph, N):
    """ 计算双层网络Graph的Pearson相关系数

    :param Graph: object |
            A MultilayerGraph object.
    :param N: int |
            Number of nodes per layer
    :return: 返回该双层网络的Pearson相关系数
    """
    # 首先判断网络是否是符合要求的网络
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."
    # 得到上下层的度numpy数组
    array1 = np.array(get_degrees_by_layer(Graph, N, 0))
    array2 = np.array(get_degrees_by_layer(Graph, N, 1))
    # 计算分子
    molecular = get_avg_degree_product(array1, array2, N) - get_avg_degree(array1, N) * get_avg_degree(array2, N)
    # 计算分母
    denominator = get_deominator_one(array1, N) * get_deominator_one(array2, N)
    return molecular / denominator

def get_sort_set_degree_layer(Graph, N, layer):
    """ 得到每层中节点度值的set集合（不重复的）

    :param Graph: object |
            A MultilayerGraph object.
    :param N: int |
            Number of nodes per layer
    :param layer: int |
            Number of layers
    :return: 返回每层中节点度值的set集合（不重复的）
    """
    # 这里为了方便后面求度值排序，每个度都加上一
    degree_layer = [i+1 for i in get_degrees_by_layer(Graph, N, layer)]
    set_degree_layer = set(degree_layer)
    sort_set_degree_layer = np.sort(list(set_degree_layer))[::-1]
    return sort_set_degree_layer

def get_sort_degree_byNode(Graph, N, layer):
    """ 得到每层中各个节点的度在所有节点度值set集合中的排名大小集合数组

    :param Graph: object |
            A MultilayerGraph object.
    :param N: int |
            Number of nodes per layer
    :param layer: int |
            Number of layers
    :return: 返回层中各个节点的度在所有节点度值set集合中的排名大小集合数组
    """
    sequ = []
    node_degree = 0
    for i in range(N):
        # 求该layer层节点i的度值
        if layer == 0:
            node_degree = Graph.degree(i)
        elif layer == 1:
            node_degree = Graph.degree(i+N)
        # 求该节点i在度的排序
        sort_set_degree_layer = get_sort_set_degree_layer(Graph, N, layer)
        for j in range(sort_set_degree_layer.size):
            if node_degree == sort_set_degree_layer[j]:
                sequ.append(j+1)
                break
    return sequ

def spearman_correlation_coefficient(Graph, N):
    """ 计算双层网络Graph的Spearman相关系数

    :param Graph: object |
            A MultilayerGraph object.
    :param N: int |
            Number of nodes per layer
    :return: 返回该双层网络的Spearman相关系数
    """
    # 首先判断网络是否是符合要求的网络
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."
    # 首先得到两个网络中各节点的排名
    array1 = np.array(get_sort_degree_byNode(Graph, N, layer=0))
    array2 = np.array(get_sort_degree_byNode(Graph, N, layer=1))
    # 求分子
    molecular = get_avg_degree_product(array1, array2, N) - get_avg_degree(array1, N) * get_avg_degree(array2, N)
    # 求分母
    denominator = get_deominator_one(array1, N) * get_deominator_one(array2, N)
    # 求Spearman相关系数
    return molecular / denominator

def overall_of_overlap_network(Graph, N):
    """ 计算双层网络整体重叠程度

    :param Graph: object |
            A MultilayerGraph object.
    :param N: int |
            Number of nodes per layer
    :return: 返回双层网络整体重叠程度 [0,1] 之间
    """
    # 首先判断网络是否是符合要求的网络
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."
    # 得到每层的边集合
    even_edge_count = 0
    edges_layer0 = Graph.get_intra_layer_edges_of_layer(layer=0)
    edges_layer1 = Graph.get_intra_layer_edges_of_layer(layer=1)

    # 找到重叠边的个数
    for i in range(len(edges_layer0)):
        temp_edge = edges_layer0[i]
        for j in range(len(edges_layer1)):
            if (temp_edge[0] + N, temp_edge[1] + N) == edges_layer1[j]:
                even_edge_count += 1
                break

    # 计算拥有连边数最少的层，得到其连边数
    edges_layer0_count = len(edges_layer0)
    edges_layer1_count = len(edges_layer1)
    min_edges_layer = edges_layer0_count if edges_layer0_count < edges_layer1_count else edges_layer1_count

    # 计算整体重叠程度，并返回
    degree_of_overlap = even_edge_count / min_edges_layer
    return degree_of_overlap


def local_of_overlap_network(Graph, node, N):
    """ 计算双层网络节点node的局部重叠程度

    :param Graph: object |
            A MultilayerGraph object
    :param node: int |
            A node
    :param N: int |
            Number of nodes per layer
    :return: 返回双层网络节点node的局部重叠程度 [0,1] 之间
    """
    # 求得节点node在每一层的邻居节点
    neighbors_layer0 = list(Graph.neighbors(node))[:-1]
    neighbors_layer1 = list(Graph.neighbors(node + N))[:-1]

    # 计算节点node共用一个邻居节点的数量
    common_nodes_count = len(list(set([i + N for i in neighbors_layer0]).intersection(set(neighbors_layer1))))

    # 计算双层网络节点node的局部重叠程度并返回
    local_of_overlap = common_nodes_count / (len(neighbors_layer0) + len(neighbors_layer1) - common_nodes_count)
    return local_of_overlap