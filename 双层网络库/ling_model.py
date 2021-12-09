# -*- coding: utf-8 -*-
"""
Created on Oct 26 2020
revised on
@author:  xiaozhi
"""
import multinetx as mx
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import seaborn as sns
import unweight_null_model as unm
import time
import random
import math
import scipy
from collections import defaultdict
import igraph as ig


def create_MultilayerGraph(N, p_g1=0.1, p_g2=0.1, seed_g1=None, seed_g2=None, intra_layer_edges_weight=None,
                           inter_layer_edges_weight=None):
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
    # adj_block[0:  N, N:2 * N] = np.identity(N)
    adj_block[:N, N:] = np.identity(N)
    adj_block += adj_block.T

    # 生成多层复杂网络对象实例
    mg = mx.MultilayerGraph(list_of_layers=[g1, g2], inter_adjacency_matrix=adj_block)
    # 设置层内以及层间边缘权重
    mg.set_edges_weights(intra_layer_edges_weight=intra_layer_edges_weight,
                         inter_layer_edges_weight=inter_layer_edges_weight)
    return mg


def create_MultilayerGraphFromPandasEdgeList(df1, df2, source_df1='source', target_df1='target', edge_attr_df1=None,
                                             create_using_df1=None, edge_key_df1=None, source_df2='source',
                                             target_df2='target',
                                             edge_attr_df2=None, create_using_df2=None, edge_key_df2=None):
    """ 只适用于两层节点数相同的双层网络，如果不同请改用单层networkx。该问题不可解决，本作者没有能力去修改multiNetx库 """
    # 创建单层网络
    g1 = mx.from_pandas_edgelist(df1, source_df1, target_df1, edge_attr_df1, create_using_df1, edge_key_df1)
    g2 = mx.from_pandas_edgelist(df2, source_df2, target_df2, edge_attr_df2, create_using_df2, edge_key_df2)
    # g1.edges =
    N = len(list(g1.nodes))

    # 创建所有随机网络对应的超矩阵
    adj_block = mx.lil_matrix(np.zeros((N * 2, N * 2)))
    # adj_block[0:  N, N:2 * N] = np.identity(N)
    adj_block[:N, N:] = np.identity(N)
    adj_block += adj_block.T

    # 生成多层复杂网络对象实例
    mg = mx.MultilayerGraph(list_of_layers=[g1, g2], inter_adjacency_matrix=adj_block)

    # # 移除g1网络不存在的边
    # g1_edges = g1.edges
    # mg_edges_0 = get_edge_0(mg, 0)
    # print(len(g1_edges))
    # print(len(mg_edges_0))
    # for v, u in mg_edges_0:
    #     if ((v,u) not in g1_edges) and ((u,v) not in g1_edges):
    #         mg.remove_edge(u,v)

    # 设置层内以及层间边缘权重
    mg.set_edges_weights(intra_layer_edges_weight=edge_attr_df1,
                         inter_layer_edges_weight=edge_attr_df1)
    return mg


# 创建单层网络
def creat_Graph(df):
    G = nx.from_pandas_edgelist(df, 'node1', 'node2', create_using=nx.Graph())
    return G


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
    # edges_up = Graph.get_intra_layer_edges_of_layer(layer=0)
    edges_up = get_edge_0(Graph, analy_layer=0)
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
            # print(f'{u}---{v}')
            # print(f'{x}---{y}')
            print(f'已成功交换{swapcount}次')
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
    keys, degrees = random_nodes_degrees(tem_keys[analy_layer * N:(analy_layer + 1) * N],
                                         tem_degrees[analy_layer * N:(analy_layer + 1) * N])
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
                print(f'已成功交换{swapcount}次')
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
                    print(f'已成功交换{swapcount}次')
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
                    D = dict_degree_nodes(degree_node_list, N)  # 找到每个度对应的所有节点，具体形式为

                    j = 0
                    for i in range(len(D)):
                        # 计算节点度为D[i]下所有节点的平均聚类系数
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
                    print(f'已成功交换{swapcount}次')
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
                    node_list = [e for e in node_list if e < N]  # 去除下层节点
                    node_list = list(set(node_list))  # 去重
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
                    print(f'已成功交换{swapcount}次')
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
        nodes = list(Graph.nodes)[analy_layer * N: (analy_layer + 1) * N]
        x_node = random.choice(nodes)
        y_node = random.choice(nodes)

        if x_node == y_node:
            continue
        # 两个节点进行交换
        # print(x_node, y_node)
        swap_one_node(Graph, x_node, y_node)
        # 加一
        swapcount += 1

    return Graph


def matchEffect_0_model(Graph, N, nswap=1, max_tries=200000, pattern='positive', analy_layer=0):
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

    swapcount = 0  # 已尝试交换节点并成功的次数
    n = 0
    # 计时
    nodes = list(Graph.nodes)[:N]

    while swapcount < nswap:
        if n >= max_tries:
            print(f'尝试次数 {n} 已超过允许的最大尝试次数，而交换成功的次数为{swapcount}')
            break

        # 随机取两个节点
        n += 1
        x_node, y_node = random.sample(nodes, 2)
        x_node_up_degree, y_node_up_degree = Graph.degree(x_node), Graph.degree(y_node)

        if x_node_up_degree == y_node_up_degree:
            continue

        x_node_down_degree, y_node_down_degree = Graph.degree(x_node + N), Graph.degree(y_node + N)
        # 判断下层，是否也是这样，如果是，则跳出，继续下一次，如果不是，则交换
        # 先判断是正匹配还是负匹配模式
        if pattern == 'positive':
            if (x_node_up_degree - y_node_up_degree) * (x_node_down_degree - y_node_down_degree) >= 0:
                continue
        elif pattern == 'negative':
            if (x_node_up_degree - y_node_up_degree) * (x_node_down_degree - y_node_down_degree) <= 0:
                continue
        # 交换
        # 这里为什么还是原始网络，因为深度复制的网络会报错，具体报错细节请自行运行好吧，
        swap_one_node(Graph, x_node, y_node, analy_layer=0)
        print(x_node, y_node)
        swapcount += 1
        print(f'有效交换 {swapcount}次')

    return Graph


def isRichClubTwo_0_model(Graph, N, k1, k2, rich=True, nswap=1, max_tries=100):
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."

    k1 += 1
    k2 += 1
    # 复制网络，并定义尝试重连次数和成功交换次数
    n = 0
    swapcount = 0
    # Graph
    up_nodes = list(Graph.nodes)[0:N]
    time_start = int(time.time())
    while swapcount < nswap:
        time_end = int(time.time())
        if time_end - time_start > 60:
            print(f'尝试次数 {n} 已超过超时，而交换成功的次数为{swapcount}')
            break
        if n >= max_tries:
            print(f'尝试次数 {n} 已超过允许的最大尝试次数，而交换成功的次数为{swapcount}')
            break
        n += 1
        # 从上层随机取两个节点，度大的则为富节点，之后在下层也找到这两个，如果两个度值相反，则让上层的节点交换位置。

        # downNodes = list(Graph.nodes)[N:]
        x_upNode, y_upNode = random.sample(up_nodes, 2)

        x_upNode_degree = Graph.degree(x_upNode)
        y_upNode_degree = Graph.degree(y_upNode)

        if x_upNode == y_upNode or x_upNode_degree == y_upNode_degree:
            continue

        # 从另一层找到对应的两个节点
        x_downNode_degree = Graph.degree(x_upNode + N)
        y_downNode_degree = Graph.degree(y_upNode + N)

        if x_downNode_degree == y_downNode_degree:
            continue

        # 如果要创建得到具有富人俱乐部特性的零模型
        if rich:
            # # 如果两层对应节点都是度大或者度小节点就跳出
            # if x_upNode_degree > y_upNode_degree and x_downNode_degree > y_downNode_degree:
            #     continue
            # if x_upNode_degree < y_upNode_degree and x_downNode_degree < y_downNode_degree:
            #     continue
            if x_upNode_degree > k1 and y_upNode_degree <= k1 and x_downNode_degree <= k2 and y_downNode_degree > k2:
                swap_one_node(Graph, x_upNode, y_upNode)
                swapcount += 1
                time_start = int(time.time())
                print(x_upNode, y_upNode)
                print(f'有效交换 {swapcount}次')
            if x_upNode_degree <= k1 and y_upNode_degree > k1 and x_downNode_degree > k2 and y_downNode_degree <= k2:
                swap_one_node(Graph, x_upNode, y_upNode)
                print(x_upNode, y_upNode)
                time_start = int(time.time())
                swapcount += 1
                print(f'有效交换 {swapcount}次')
        else:
            # if x_upNode_degree > y_upNode_degree and x_downNode_degree < y_downNode_degree:
            #     continue
            # if x_upNode_degree < y_upNode_degree and x_downNode_degree > y_downNode_degree:
            #     continue
            if x_upNode_degree > k1 and y_upNode_degree <= k1 and x_downNode_degree > k2 and y_downNode_degree <= k2:
                swap_one_node(Graph, x_upNode, y_upNode)
                time_start = int(time.time())
                print(x_upNode, y_upNode)
                swapcount += 1
                print(f'有效交换 {swapcount}次')
            if x_upNode_degree <= k1 and y_upNode_degree > k1 and x_downNode_degree <= k2 and y_downNode_degree > k2:
                swap_one_node(Graph, x_upNode, y_upNode)
                time_start = int(time.time())
                print(x_upNode, y_upNode)
                swapcount += 1
                print(f'有效交换 {swapcount}次')
        # 否则就交换上层节点
    return Graph

def mesoscale_changenode_model_cengjian(Graph, str_layer1, str_layer2, nswap=1, analy_layer=0):
    swapcount = 0  # 已尝试交换节点并成功的次数
    # 基于这些连边使用igraph创建一个新网络
    if analy_layer == 0:
        G_nodes = get_Graph_nodes(str_layer1,' ', 0)
        Gi = ig.Graph.Read_Edgelist(str_layer1)
    else:
        G_nodes = get_Graph_nodes(str_layer2,' ', 0)
        Gi = ig.Graph.Read_Edgelist(str_layer2)

    G = Gi.as_undirected()        # 检测到的社区+
    community_list = G.community_multilevel(weights=None, return_levels=False)

    # 提取真正的社区集合
    community_true = []

    for i in range(len(community_list)):
        temp_list = community_list[i]
        if (len(temp_list) < 2) and (temp_list[0] not in G_nodes):
            continue
        community_true.append(temp_list)

        # 随机选择一层中的两个社区并选分别选择一个节点进行交换

    # 得到社区的下标集合
    community_true_index = [i for i in range(len(community_true))]
    while swapcount < nswap:

        t1, t2 = random.sample(community_true_index, 2)
        # 取出两个点
        x_node = random.choice(community_true[t1])
        y_node = random.choice(community_true[t2])
        #         print(community_true)
        #         print(f'{t1}---{t2}')
        # print(f'{x_node}---{y_node}')
        # 随机选择两个社区的节点进行交换
        if x_node == y_node:
            continue
        # 两个节点进行交换
        # 其实也就是交换两个节点所连接的边

        # 找到节点 x and y 的分别边集合
        edges_all = Graph.edges
        # 先找到节点x y的所有边
        x_node_edges_node = []
        y_node_edges_node = []

        for v, u in edges_all:
            if x_node == v and y_node != u:
                x_node_edges_node.append(u)
            elif x_node == u and y_node != v:
                x_node_edges_node.append(v)
            elif y_node == v and x_node != u:
                y_node_edges_node.append(u)
            elif y_node == u and x_node != v:
                y_node_edges_node.append(v)

        # print(x_node_edges_node)
        # print(y_node_edges_node)
        # 删除两个邻居节点列表中相同的元素
        x_node_edges_node_new = [i for i in x_node_edges_node if i not in y_node_edges_node]

        y_node_edges_node_new = [i for i in y_node_edges_node if i not in x_node_edges_node]

        # print('---------------------')
        # print(x_node_edges_node_new)
        # print(y_node_edges_node_new)
        # 交换边
        if len(x_node_edges_node_new) > 0:
            for node in x_node_edges_node_new:
                Graph.remove_edge(x_node, node)
                Graph.add_edge(y_node, node)
        if len(y_node_edges_node_new) > 0:
            for node in y_node_edges_node_new:
                Graph.remove_edge(node, y_node)
                Graph.add_edge(node, x_node)

        # 加一
        swapcount += 1
        # print(lm.get_edge_0(Graph, 0))
        # 交换完毕后，改变社区集合节点的位置
        community_true[t1].remove(x_node)
        community_true[t1].append(y_node)
        community_true[t2].remove(y_node)
        community_true[t2].append(x_node)

    return Graph

def mesoscale_changenode_model_cengnei(Graph, str_layer1, str_layer2, nswap=1, analy_layer=0):
    swapcount = 0  # 已尝试交换节点并成功的次数
    # 基于这些连边使用igraph创建一个新网络
    if analy_layer == 0:
        G_nodes = get_Graph_nodes(str_layer1,sep=' ', has_name=0)
        Gi = ig.Graph.Read_Edgelist(str_layer1)
    else:
        G_nodes = get_Graph_nodes(str_layer2,' ', 0)
        Gi = ig.Graph.Read_Edgelist(str_layer2)

    G = Gi.as_undirected()        # 检测到的社区+
    community_list = G.community_multilevel(weights=None, return_levels=False)
    # 提取真正的社区集合
    community_true = []

    for i in range(len(community_list)):
        temp_list = community_list[i]
        if (len(temp_list) < 2) and (temp_list[0] not in G_nodes):
            continue
        community_true.append(temp_list)

        # 随机选择一层中的两个社区并选分别选择一个节点进行交换

    # 得到社区的下标集合
    community_true_index = [i for i in range(len(community_true))]
    while swapcount < nswap:

        t1= random.choice(community_true_index)
        # 取出两个点
        x_node,y_node = random.sample(community_true[t1], 2)
        #         print(community_true)
        # # print(f'{t1}---{t2}')
        # print(f'{x_node}---{y_node}')
        # 随机选择两个社区的节点进行交换
        if x_node == y_node:
            continue
        # 两个节点进行交换
        # 其实也就是交换两个节点所连接的边

        # 找到节点 x and y 的分别边集合
        edges_all = Graph.edges
        # 先找到节点x y的所有边
        x_node_edges_node = []
        y_node_edges_node = []

        for v,u in edges_all:
            if x_node == v and y_node != u:
                x_node_edges_node.append(u)
            elif x_node == u and y_node != v:
                x_node_edges_node.append(v)
            elif y_node == v and x_node != u:
                y_node_edges_node.append(u)
            elif y_node == u and x_node != v:
                y_node_edges_node.append(v)

        # print(x_node_edges_node)
        # print(y_node_edges_node)
        # 删除两个邻居节点列表中相同的元素
        x_node_edges_node_new = [i for i in x_node_edges_node if i not in y_node_edges_node]

        y_node_edges_node_new = [i for i in y_node_edges_node if i not in x_node_edges_node]

        # print('---------------------')
        # print(x_node_edges_node_new)
        # print(y_node_edges_node_new)
        # 交换边
        if len(x_node_edges_node_new) > 0:
            for node in x_node_edges_node_new:
                Graph.remove_edge(x_node, node)
                Graph.add_edge(y_node, node)
        if len(y_node_edges_node_new) > 0:
            for node in y_node_edges_node_new:
                Graph.remove_edge(node, y_node)
                Graph.add_edge(node, x_node)
        # 加一
        swapcount += 1

    return Graph

def mesoscale_changenode_model_random(Graph, nswap=1):
    swapcount = 0  # 已尝试交换节点并成功的次数
    nodes = Graph.nodes
    while swapcount < nswap:
        # 取出两个点
        x_node,y_node = random.sample(nodes, 2)
        # 随机选择两个社区的节点进行交换
        if x_node == y_node:
            continue
        # print(f'{x_node}---{y_node}')
        # 两个节点进行交换
        # 其实也就是交换两个节点所连接的边
        # 找到节点 x and y 的分别边集合
        edges_all = Graph.edges
        # 先找到节点x y的所有边
        x_node_edges_node = []
        y_node_edges_node = []

        for v,u in edges_all:
            if x_node == v and y_node != u:
                x_node_edges_node.append(u)
            elif x_node == u and y_node != v:
                x_node_edges_node.append(v)
            elif y_node == v and x_node != u:
                y_node_edges_node.append(u)
            elif y_node == u and x_node != v:
                y_node_edges_node.append(v)

        # 删除两个邻居节点列表中相同的元素
        x_node_edges_node_new = [i for i in x_node_edges_node if i not in y_node_edges_node]
        y_node_edges_node_new = [i for i in y_node_edges_node if i not in x_node_edges_node]
        # 交换边
        if len(x_node_edges_node_new) > 0:
            for node in x_node_edges_node_new:
                Graph.remove_edge(x_node, node)
                Graph.add_edge(y_node, node)
        if len(y_node_edges_node_new) > 0:
            for node in y_node_edges_node_new:
                Graph.remove_edge(node, y_node)
                Graph.add_edge(node, x_node)
        # 加一
        swapcount += 1

    return Graph


def get_edges(Graph, N):
    """
    已经弃用，有问题。直接返回layer = 0 底层的边
    :param Graph:
    :param N:
    :return:
    """
    edges = []
    for i in range(N):
        # 找出本层所有相邻节点，不包含其它层的
        neigh_nodes = list(Graph[i])
        # 对列表进行排序
        neigh_nodes.sort()
        neigh_nodes = neigh_nodes[:-1]
        # 减去之前以重复的节点
        neigh_node = [node for node in neigh_nodes if i < node]
        # 以元组形式返回
        for node in neigh_node:
            edges.append((i, node))
    return edges


# 将双层网络对象转换为csv文件，仅仅是把第一层也就是交换的哪一层导出
def multiGraph_to_csv(Graph, analy_layer=0):
    N = int(len(Graph.nodes) / 2)
    richedges0 = get_edge_0(Graph, analy_layer)
    list_n1, list_n2 = [], []
    for i in range(len(richedges0)):
        if analy_layer == 1:
            list_n1.append(richedges0[i][0] - N)
            list_n2.append(richedges0[i][1] - N)
        else:
            list_n1.append(richedges0[i][0])
            list_n2.append(richedges0[i][1])
    df = pd.DataFrame(({'node1': list_n1, 'node2': list_n2}))
    return df


# 将单层网络对象转换为csv文件
def graph_to_csv(G):
    list_n1, list_n2 = [], []
    for i in range(len(G.edges())):
        list_n1.append(list(G.edges())[i][0])
        list_n2.append(list(G.edges())[i][1])
    df = pd.DataFrame(({'node1': list_n1, 'node2': list_n2}))
    return df


# 创建具有或者不具有富人俱乐部特性的  单层零模型网络以及返回其富人俱乐部特征相关系数
def isRichClubSingle_0_model(Graph, k=1, rich=True, nswap=1, max_tries=100, connected=1):
    richG = None
    if rich:
        richG = unm.rich_club_create(Graph, k, nswap, max_tries, connected)
    else:
        richG = unm.rich_club_break(Graph, k, nswap, max_tries, connected)
    return richG


def isMatchEffectSingle_0_model(Graph, k=1, pattern='positive', nswap=1, max_tries=100, connected=1):
    matchG = None
    if pattern == 'positive':
        matchG = unm.assort_mixing(Graph, k, nswap, max_tries, connected)
    else:
        matchG = unm.disassort_mixing(Graph, k, nswap, max_tries, connected)
    return matchG


# 双通节点
def double_nodes(G1, G2):
    return list(set(G1.nodes()).intersection(set(G2.nodes())))


# 求单层连通性
def is_connect_single(Graph):
    return nx.is_connected(Graph)


# 平均度
def avg_degree_single(Graph):
    degrees = [i[1] for i in list(Graph.degree())]
    return sum(degrees) / len(degrees)


# 平均最短路径
def average_shorted_pathLength_single(Graph):
    return nx.average_shortest_path_length(Graph)


# 层内度相关系数
def dac_single(G):
    return nx.degree_assortativity_coefficient(G)


def swap_one_node(Graph, node_1, node_2, analy_layer=0):
    """
    把 analy_layer 这个层中的两个节点进行置换，保持网络结构不变
    算法过程也就是通过节点的连边来进行交换节点（因为双层网络库的原因，只能如此做）
    (node_1, node_2)
    :param Graph: object
                a MultilayerGraph object.
    :param node_1: node 1 （int）
    :param node_2: node 2 （int）
    :param analy_layer: int (0 | 1)
                    choose analy layer.

    :return: None
    """
    swapped_edges = []  # 已交换边的集合
    edges = get_edge_0(Graph=Graph, analy_layer=analy_layer)
    for item in edges:
        if item.count(node_1) == 0 and item.count(node_2) == 0:
            continue
        elif item.count(node_1) == 1 and item.count(node_2) == 1:
            continue
        elif item.count(node_1) == 1 and item.count(node_2) == 0:
            v, w = item
            if (v, w) not in swapped_edges and (w, v) not in swapped_edges:
                Graph.remove_edge(v, w)
            tem_n = w if v == node_1 else v
            Graph.add_edge(node_2, tem_n)
            swapped_edges.append((node_2, tem_n))
        elif item.count(node_1) == 0 and item.count(node_2) == 1:
            v, w = item
            if (v, w) not in swapped_edges and (w, v) not in swapped_edges:
                Graph.remove_edge(v, w)
            tem_n = w if v == node_2 else v
            Graph.add_edge(node_1, tem_n)
            swapped_edges.append((node_1, tem_n))


def dict_degree_nodes(degree_node_list, N):
    """返回的字典为{度：[节点1，节点2，..], ...}，其中节点1和节点2有相同的度"""
    D = {}
    for degree_node_i in degree_node_list:
        if degree_node_i[1] > N:
            continue
        if degree_node_i[0] not in D:
            D[degree_node_i[0]] = [degree_node_i[1]]
        else:
            if degree_node_i[1] in D[degree_node_i[0]]:
                continue
            D[degree_node_i[0]].append(degree_node_i[1])
    return D


def sub_edgesBigAndSmall(edges_list1, edges_list2):
    """两个边列表做差集"""
    # sub_edges = []
    edges_list22 = [(y, x) for x, y in edges_list2]
    sub_edges = list(set(edges_list1) - set(edges_list22))
    # edges_all = edges_list1 + edges_list2

    # 对edges_list1, edges_list2
    # for x,y in edges_all:
    #     if ((x,y) in edges_list1 or (y,x) in edges_list1) and ((x,y) in edges_list2 or (y,x) in edges_list2):
    #         continue
    #     else:
    #         sub_edges.append((x,y))
    return sub_edges


def get_edge_0(Graph, analy_layer):
    """1阶零模型算法中求每次断边重连时第0层的边（时时变化的只有get_intra_layer_edges()总边）
        当analy_layer = 1时，返回的节点标号是加上 N 的
        analy_layer : int (0 | 1)
            choose analy layer.
        """
    if Graph.get_intra_layer_edges() == []:
        return '该网络非原始网络，而是经过Copy函数复制的函数，请修改为原始网络对象'
    if analy_layer == 1:
        return list(Graph.get_intra_layer_edges_of_layer(1))
    ### 老是这个问题，有的时候好有的时候坏
    intra_edges = list(Graph.get_intra_layer_edges())
    inter_edges = list(Graph.get_inter_layer_edges())
    # 搞掉层间连线
    sub_edges = sub_edgesBigAndSmall(intra_edges, inter_edges)
    sub_edges_one = list(set(sub_edges) - set(list(Graph.get_intra_layer_edges_of_layer(1))))
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
    matrix_data = mx.adjacency_matrix(Graph, weight=weight).todense()
    assert int(matrix_data.shape[0] / 2) == N, \
        "传入的矩阵中单层节点个数要与传入的N节点相等"
    frontff_dat = f"""DL\nN={N} NM=2\nFORMAT = FULLMATRIX DIAGONAL PRESENT\nDATA:\n"""
    arr_up = matrix_data.getA()[N:2 * N, N:2 * N]
    arr_down = matrix_data.getA()[0:N, 0:N]
    arr_data = np.vstack((arr_down, arr_up))
    np.savetxt(fname, arr_data, fmt='%s', delimiter=' ')
    file_name = fname
    with open(file_name, 'r') as f:
        lines = f.readlines()
    with open(file_name, 'w') as n:
        n.write(frontff_dat)
        n.writelines(lines)


# 计算相关系数
# 求该双层网络富人俱乐部的系数
def rcValue(G, m, n):
    '''
    :param G: 网络
    :param m: 进行置换网络的富节点的度值
    :param n: 不进行置换网络的富节点的度值
    :return: 返回该网络富人俱乐部的系数
    '''
    nodes = list(G.nodes)
    N = int(len(G.nodes) / 2)
    hub = [e for e in nodes[0:N] if G.degree(e) > m + 1]
    hubs2 = [e for e in nodes[N:] if G.degree(e) > n + 1]
    hubs1 = [e + N for e in hub]
    bj = list(set(hubs1).union(set(hubs2)))
    jj = list(set(hubs1).intersection(set(hubs2)))
    return len(jj) / len(bj)


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
    return Graph.degree(N * layer + node) - 1


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
    return np.sum(array1 * array2) / N


def get_avg_degree(array, N):
    """ 计算一层网络的平均度

    :param array: numpy.ndarray |
            A numpy array
    :param N: int |
            Number of nodes per layer
    :return: Average degree of the network
    """
    return np.sum(array) / N


def get_avg_degree_bylayer(Graph, N, layer=0):
    """
    得到一层网络的平均度
    :param Graph: 双层网络
    :param analy_layer: 哪一层
    :return: 度值
    """
    degrees = [i[1] for i in list(Graph.degree())[layer * N: (layer + 1) * N - 1]]
    average_degree = sum(degrees) / len(degrees)
    return average_degree


def get_avg_degree(array, N):
    """
    通过度的array数组来求该层网络的平均度
    :param array:
    :param N:
    :return:
    """
    return sum(list(array)) / N


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
    degree_layer = [i + 1 for i in get_degrees_by_layer(Graph, N, layer)]
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
            node_degree = Graph.degree(i + N)
        # 求该节点i在度的排序
        sort_set_degree_layer = get_sort_set_degree_layer(Graph, N, layer)
        for j in range(sort_set_degree_layer.size):
            if node_degree == sort_set_degree_layer[j]:
                sequ.append(j + 1)
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
    # edges_layer0 = Graph.get_intra_layer_edges_of_layer(layer=0)
    edges_layer0 = get_edge_0(Graph, 0)
    edges_layer1 = Graph.get_intra_layer_edges_of_layer(layer=1)

    # 找到重叠边的个数
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if ((i, j) in edges_layer0 or (j, i) in (edges_layer0)) and ((i + N, j + N) in edges_layer1 or (j + N, i + N) in (edges_layer1)):
                even_edge_count += 1
    # for i in range(len(edges_layer0)):
    #     temp_edge = edges_layer0[i]
    #     for j in range(len(edges_layer1)):
    #         if (temp_edge[0] + N, temp_edge[1] + N) == edges_layer1[j]:
    #             even_edge_count += 1
    #             break

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


"""
 长向量的统计量
 注意：可视化时我把0层也就是α层与β层交换了位置，放到了上层。目的是为了在我们对α层进行零模型构建时观察的方便！！
 在求多链路或多度时，注意α层还是前 N 个节点，属于第 0 层。
"""


def get_adjacency_matrix(Graph):
    """
    求双层网络两层的邻接矩阵
    :param Graph: 双层网络
    :return: α层邻接矩阵, α层邻接矩阵
    """
    N = int(len(Graph.nodes) / 2)
    # α层邻接矩阵
    matrix_down_a = mx.adjacency_matrix(Graph, nodelist=list(Graph.nodes)[0:N], weight=None).todense()
    # α层邻接矩阵
    matrix__top_b = mx.adjacency_matrix(Graph, nodelist=list(Graph.nodes)[N:], weight=None).todense()
    return matrix_down_a, matrix__top_b


def get_longVector(matrix):
    """
    求长向量
    :param matrix: 网络对应的邻接矩阵
    :return: 邻接矩阵对应的长向量
    """
    matrix_list = matrix.getA()
    len_matrix_list = len(matrix_list)
    V = [matrix_list[i][j] for i in range(len_matrix_list) for j in range(len_matrix_list) if i != j]
    return V


def get_TwoLongVector_by_MultiGraph(Graph):
    """
    求双层网络的两个长向量
    :param Graph: 双层网络
    :return: 对应两个单层网络的长向量
    """
    matrix_down_a, matrix__top_b = get_adjacency_matrix(Graph)
    V1 = get_longVector(matrix_down_a)
    V2 = get_longVector(matrix__top_b)
    return V1, V2


def pearson_correlation_coefficient_byLongVector(Graph):
    """ 计算双层网络Graph的Pearson相关系数

        :param Graph: object |
                A MultilayerGraph object.
        :return: 返回该双层网络的Pearson相关系数
        """
    # 首先判断网络是否是符合要求的网络
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."

    V1, V2 = get_TwoLongVector_by_MultiGraph(Graph)
    N = len(V1)  # 这里N为长向量列表的长度
    array1 = np.array(V1)
    array2 = np.array(V2)
    # 计算分子
    molecular = get_avg_degree_product(array1, array2, N) - get_avg_degree(array1, N) * get_avg_degree(array2, N)
    # 计算分母
    c1 = get_deominator_one(array1, N)
    denominator = get_deominator_one(array1, N) * get_deominator_one(array2, N)
    return molecular / denominator


def get_longVector_probability(vector):
    """
    计算长向量列表中每个元素出现的概率
    :param vector: 长向量 list
    :return: 对应概率列表
    """
    len_vector = len(vector)
    set_vector = set(vector)
    # 把值对应的概率以字典的方式搞出来
    dict_vector_p = {}
    for v in set_vector:
        # dict_vector_p[v] = round(vector.count(v) / len_vector, 2) # 保不保留小数看自己了
        dict_vector_p[v] = vector.count(v) / len_vector
    # probability = [dict_vector_p[v] for v in vector]
    probability = list(dict_vector_p.values())
    return probability


def get_information_entropy(V_probability):
    """
    求单个长向量对应的信息熵
    :param V_probability: 长向量对应的概率集合
    :return: 信息熵值
    """
    H_V = 0
    for p in V_probability:
        p_vlaue = p * math.log(p, 2)
        H_V += p_vlaue
    return -H_V


def mutual_information_byLongVector(Graph):
    """
    求长向量的互信息
    :param Graph: 双层网络
    :return: 互信息值
    """
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."
    # 得到两层网络对应的长向量 V1对应的是下层网络[0:N] V2对应的是上层网络[N:2N]
    V1, V2 = get_TwoLongVector_by_MultiGraph(Graph)
    V1_probability = get_longVector_probability(V1)
    V2_probability = get_longVector_probability(V2)
    # 声明对应的变量
    H_X = get_information_entropy(V1_probability)
    H_Y = get_information_entropy(V2_probability)
    # 联合信源对应的信息熵
    V1_2 = [x * y for x in V1 for y in V2]
    V1_2_probability = get_longVector_probability(V1_2)
    H_XY = get_information_entropy(V1_2_probability)

    # 互信息的值
    I = H_X + H_Y - H_XY
    return I


def multi_link(Graph, node1, node2):
    """
    求节点node1 与节点node2 之间的多链路
    注意：可视化时我把0层也就是α层与β层交换了位置，放到了上层。目的是为了在我们对α层进行零模型构建时观察的方便！！
    在求多链路或多度时，注意α层还是前 N 个节点，属于第 0 层。
    :param Graph: 双层网络
    :param node1: 节点1 值为[0, N) 左闭右开
    :param node2: 节点2 值为[0, N) 左闭右开
    :return: 返回对应的多链路
    """
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."

    N = int(len(Graph.nodes) / 2)
    is_link_layer_a = node2 in Graph[node1]
    is_link_layer_b = (node2 + N) in Graph[node1 + N]
    a = 1 if is_link_layer_a else 0
    b = 1 if is_link_layer_b else 0
    return (a, b)


def multi_degree(Graph, node):
    """
    求节点node 的多度
    :param Graph: 双层网络
    :param node: 节点node 值为[0, N) 左闭右开  如果节点1则对应 node=0【注意！！！】【统一要求！】
    :return: 返回其节点node的三个多度值
    """
    # 两层的邻接矩阵
    matrix_down_a, matrix__top_b = get_adjacency_matrix(Graph)
    array_down_a = np.array(matrix_down_a)
    array__top_b = np.array(matrix__top_b)
    K_a = np.sum(array_down_a[node] * (1 - array__top_b[node]))
    K_b = np.sum(array_down_a[node] * (1 - array_down_a[node]))
    K_ab = np.sum(array_down_a[node] * array__top_b[node])
    return K_a, K_b, K_ab


def get_degree_ab_nodeNum(Graph, N, degree_a, degree_b):
    """
    返回α层度值为degree_a，β层度值为degree_b的节点个数
    :param Graph:
    :param N:
    :param degree_a:
    :param degree_b:
    :return: int
    """
    res = 0
    dict_degree = dict(Graph.degree())
    for i in range(N):
        if dict_degree[i] == degree_a and dict_degree[i + N] == degree_b:
            res += 1
    return res


def huizhi_geshi_heatmap():
    sns.set(style="darkgrid", palette="muted")
    sns.set_context("notebook", font_scale=0.8)
    plt.figure(figsize=(6, 5), dpi=144, facecolor=None, edgecolor=None, frameon=False)
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 调节字体显示为中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    plt.rcParams['font.family'] = 'sans-serif'


def degree_feature_matrix(Graph):
    """
    求度特征矩阵，以热力图的形式呈现出来
    如果热力图呈正相关则该双层网络为正相关网络，否则负相关或者无相关性
    图的横轴为α层，从度最小到最大
    图的纵轴为β层，从度最小到最大
    每一个网格分别是两层网络度值分别（kα，kβ）的节点个数值
    :param Graph: 双层网络对象
    :return: 无返回值
    """
    # 先将两层网络的度值从大到小排列
    # a是下层  b是上层
    N = int(len(Graph.nodes) / 2)
    degree_as, degree_bs = list(set(get_degrees_by_layer(Graph, N, layer=0))), list(
        set(get_degrees_by_layer(Graph, N, layer=1)))
    degree_as.sort()  # 对列表本身进行排序
    degree_bs.sort()
    #     degree_bs = degree_bs[::-1]
    # 排序后生成二维数组，从
    # 再找对应每个（kα，kβ）的节点个数
    degree_ab_nodesNum = [(degree_a, degree_b, get_degree_ab_nodeNum(Graph, N, degree_a + 1, degree_b + 1)) for degree_a
                          in degree_as for degree_b in degree_bs]

    # 从图的左下角开始，也就是plt.cm.Blues
    huizhi_geshi_heatmap()
    df = pd.DataFrame(degree_ab_nodesNum, columns=['a', 'b', 'val'])
    target = pd.pivot_table(df, values='val', index='b', columns='a')
    ax = sns.heatmap(target, cmap=plt.cm.Blues, linewidths=1, annot=False)
    # 因为是在第四象限，所以随着X和Y轴的增大为正相关，随着X轴增大Y减小为负相关
    plt.xlabel('α层', fontsize=15)
    plt.ylabel('β层', fontsize=15)
    #     plt.savefig("./1.png")
    plt.show()


def remain_average_degree(Graph, analy_layer, degree):
    """
    求度相关下的余平均度，根据analy_layer来确定分析的是Kα还是Kβ
    degree是另一层度值为degree
    :param Graph: 双层网络对象
    :param analy_layer: 分析的网络层
    :param degree: 所分析的度值
    :return:
    """
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."
    assert (degree + 1 in dict(Graph.degree()).values()) == True, \
        f"抱歉，该网络中不存在度值为{degree}的节点"
    N = int(len(Graph.nodes) / 2)
    degree_as, degree_bs = list(set(get_degrees_by_layer(Graph, N, layer=0))), list(
        set(get_degrees_by_layer(Graph, N, layer=1)))
    degree_as.sort()  # 对列表本身进行排序
    degree_bs.sort()
    if analy_layer == 0:
        degree_ab_nodesNum = [get_degree_ab_nodeNum(Graph, N, degree_a + 1, degree + 1) for degree_a in degree_as]
        degree_multi_nodes_Num = np.array(degree_as) * np.array(degree_ab_nodesNum)
        if np.sum(degree_ab_nodesNum) == 0:
            return 0
        return np.sum(degree_multi_nodes_Num) / np.sum(degree_ab_nodesNum)
    if analy_layer == 1:
        degree_ab_nodesNum = [get_degree_ab_nodeNum(Graph, N, degree_b + 1, degree + 1) for degree_b in degree_bs]
        degree_multi_nodes_Num = np.array(degree_bs) * np.array(degree_ab_nodesNum)
        if np.sum(degree_ab_nodesNum) == 0:
            return 0
        return np.sum(degree_multi_nodes_Num) / np.sum(degree_ab_nodesNum)


def hiuzhi_remain_average_degree(Graph, analy_layer, nswap=20000, max_tries=2000000):
    """
    为了更好看清其度相关的变化，利用可视化进行搞定
    :param Graph: 双层网络对象
    :param analy_layer: 求哪一层的余平均度
    :param degrees: 另一层的度值列表
    :param res1: 原始网络的余平均度序列
    :param res2: 度正相关网络的余平均度序列
    :param res3: 度负相关网络的余平均度序列
    :return: 无返回值
    """
    N = int(len(Graph.nodes) / 2)
    degree_s = list(set(get_degrees_by_layer(Graph, N, layer=1 if analy_layer == 0 else 0)))
    degree_s.sort()
    res = []
    for degree in degree_s:
        res.append((remain_average_degree(Graph, analy_layer, degree), degree, '原始网络'))
    G1 = matchEffect_0_model(Graph, N, nswap, max_tries, pattern='positive', analy_layer=analy_layer)
    for degree in degree_s:
        res.append((remain_average_degree(G1, analy_layer, degree), degree, '度正相关零模型网络'))
    G2 = matchEffect_0_model(Graph, N, nswap, max_tries, pattern='negative', analy_layer=analy_layer)
    for degree in degree_s:
        res.append((remain_average_degree(G2, analy_layer, degree), degree, '度负相关零模型网络'))
    if len(res) == 0:
        print("余平均度计算为空")
        return
    x_degree = degree_s
    huizhi_geshi_lineplot()
    # plt.style.use('ggplot')
    plt.xlabel("度值", fontsize=10, family='SimHei')
    plt.ylabel("余平均度值", fontsize=10, family='SimHei')
    plt.xticks(x_degree, fontproperties='Times New Roman', size=10)
    plt.yticks(fontproperties='Times New Roman', size=10)

    df = pd.DataFrame(res, columns=['y', 'x', 'region'])
    sns.lineplot(x="x", y="y", hue="region", data=df)
    # plt.grid(True)
    plt.show()


def rich_degree_feature_matrix(Graph, k1, k2):
    """
    求度特征矩阵，以热力图的形式呈现出来
    如果热力图呈正相关则该双层网络为正相关网络，否则负相关或者无相关性
    图的横轴为α层，从度最小到最大
    图的纵轴为β层，从度最小到最大
    每一个网格分别是两层网络度值分别（kα，kβ）的节点个数值
    :param Graph: 双层网络对象
    :param k1: α层也就是底层的富人门限度值
    :param k2: β层也就是上层的富人门限度值
    :return: 无返回值
    """
    # 先将两层网络的度值从大到小排列
    # a是下层  b是上层
    N = int(len(Graph.nodes) / 2)
    degree_as, degree_bs = list(set(get_degrees_by_layer(Graph, N, layer=0))), list(
        set(get_degrees_by_layer(Graph, N, layer=1)))
    degree_as.sort()  # 对列表本身进行排序
    degree_bs.sort()
    degree_as = [degree for degree in degree_as if degree > k1]
    degree_bs = [degree for degree in degree_bs if degree > k2]
    # 排序后生成二维数组，从
    # 再找对应每个（kα，kβ）的节点个数
    degree_ab_nodesNum = [(degree_a, degree_b, get_degree_ab_nodeNum(Graph, N, degree_a + 1, degree_b + 1)) for degree_a
                          in degree_as for degree_b in degree_bs]

    # 从图的左下角开始，也就是
    huizhi_geshi_heatmap()
    df = pd.DataFrame(degree_ab_nodesNum, columns=['a', 'b', 'val'])
    target = pd.pivot_table(df, values='val', index='b', columns='a')
    font = {'family': 'Times New Roman', 'size': 12, 'fontstyle': 'italic'}
    vmax = 1 if df['val'].max() == 0 else df['val'].max()
    ax = sns.heatmap(target, vmin=0, vmax=vmax, cmap=plt.cm.Blues, linewidths=1, annot=False, annot_kws=font)
    # 因为是在第四象限，所以随着X和Y轴的增大为正相关，随着X轴增大Y减小为负相关
    plt.xlabel('α层', fontsize=15)
    plt.ylabel('β层', fontsize=15)
    #     plt.savefig("./1.png")
    plt.show()


def rich_remain_average_degree(Graph, analy_layer, degree, k1, k2):
    """
    求度相关下的余平均度，根据analy_layer来确定分析的是Kα还是Kβ
    degree是另一层度值为degree
    :param Graph: 双层网络对象
    :param analy_layer: 分析的网络层
    :param degree: 所分析的度值
    :param k1: α层也就是底层的富人门限度值
    :param k2: β层也就是上层的富人门限度值
    :return:
    """
    assert mx.is_connected(Graph) == True, \
        "The network must be reachable.提示：遇到这种问题，一般是生成的随机化网络不满足条件，解决办法是重新运行或者在创建双层网络时增加边出现的概率。"
    assert Graph.is_directed() == False, \
        "It is only allowed for undirected complex network."
    assert len(Graph) > 3, \
        "Graph has less than three nodes."
    assert (degree + 1 in dict(Graph.degree()).values()) == True, \
        f"抱歉，该网络中不存在度值为{degree}的节点"
    N = int(len(Graph.nodes) / 2)
    degree_as, degree_bs = list(set(get_degrees_by_layer(Graph, N, layer=0))), list(
        set(get_degrees_by_layer(Graph, N, layer=1)))
    degree_as.sort()  # 对列表本身进行排序
    degree_bs.sort()
    degree_as = [degree for degree in degree_as if degree > k1]
    degree_bs = [degree for degree in degree_bs if degree > k2]

    if analy_layer == 0:
        degree_ab_nodesNum = [get_degree_ab_nodeNum(Graph, N, degree_a + 1, degree + 1) for degree_a in degree_as]
        degree_multi_nodes_Num = np.array(degree_as) * np.array(degree_ab_nodesNum)
        if np.sum(degree_ab_nodesNum) == 0:
            return 0
        return np.sum(degree_multi_nodes_Num) / np.sum(degree_ab_nodesNum)
    if analy_layer == 1:
        degree_ab_nodesNum = [get_degree_ab_nodeNum(Graph, N, degree_b + 1, degree + 1) for degree_b in degree_bs]
        degree_multi_nodes_Num = np.array(degree_bs) * np.array(degree_ab_nodesNum)
        if np.sum(degree_ab_nodesNum) == 0:
            return 0
        return np.sum(degree_multi_nodes_Num) / np.sum(degree_ab_nodesNum)


def huizhi_geshi_lineplot():
    sns.set(style="ticks", palette="muted")
    sns.set_context("notebook", font_scale=0.8)
    plt.figure(figsize=(5.5, 4), dpi=144, facecolor=None, edgecolor=None, frameon=False)
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 调节字体显示为中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    plt.rcParams['font.family'] = 'sans-serif'


def hiuzhi_rich_remain_average_degree(Graph, analy_layer, k1, k2, nswap=20000, max_tries=2000000):
    """
    为了更好看清其度相关的变化，利用可视化进行搞定
    :param Graph: 双层网络对象
    :param analy_layer: 求哪一层的余平均度
    :param degrees: 另一层的度值列表
    :param k1: α层也就是底层的富人门限度值
    :param k2: β层也就是上层的富人门限度值
    :param res1: 原始网络的余平均度序列
    :param res2: 度正相关网络的余平均度序列
    :param res3: 度负相关网络的余平均度序列
    :return: 无返回值
    """
    N = int(len(Graph.nodes) / 2)
    degree_s = list(set(get_degrees_by_layer(Graph, N, layer=1 if analy_layer == 0 else 0)))
    degree_s.sort()
    res = []
    for degree in degree_s:
        res.append((rich_remain_average_degree(Graph, analy_layer, degree, k1, k2), degree, '原始网络'))
    G1 = isRichClubTwo_0_model(Graph, N, k1, k2, rich=True, nswap=nswap, max_tries=max_tries)
    for degree in degree_s:
        res.append((rich_remain_average_degree(G1, analy_layer, degree, k1, k2), degree, '具有富人俱乐部的零模型网络'))
    G2 = isRichClubTwo_0_model(Graph, N, k1, k2, rich=False, nswap=nswap, max_tries=max_tries)
    for degree in degree_s:
        res.append((rich_remain_average_degree(G2, analy_layer, degree, k1, k2), degree, '不具有富人俱乐部的零模型网络'))
    if len(res) == 0:
        print("余平均度计算为空")
        return

    x_degree = degree_s
    huizhi_geshi_lineplot()
    plt.xlabel("度值", fontsize=10, family='SimHei')
    plt.ylabel("余平均度值", fontsize=10, family='SimHei')
    plt.xticks(x_degree, fontproperties='Times New Roman', size=10)
    plt.yticks(fontproperties='Times New Roman', size=10)
    df = pd.DataFrame(res, columns=['y', 'x', 'region'])
    sns.lineplot(x="x", y="y", hue="region", data=df)
    plt.show()

"""
    检验方法
"""
# Z检验
def z_jianyan(origin_val, mean_val, std_val):
    return (origin_val - mean_val) / std_val


"""
   相似性算法 使用networkx网络结构（为了节点不对齐的广义多层网络）
"""

"""
首先引入了让信息更丰富表现出来的属性矩阵形式
   引入属性矩阵
"""
def get_sorted_node_byGraph(Graph):
    """
    因为networkx取得的节点是无序字符类型，这里变成了有序字符类型
    ['1','2','3','6','7','8']
    使用networkx计算单层（目的是节点不对齐的双层网络）而写
    :param Graph: networkx网络对象
    :return: 节点没有顺序的度列表（也就是节点并不是从小到大，而是随机的）
    """
    str_node_list = nx.nodes(Graph)
    int_node_list = [int(n) for n in str_node_list]
    int_node_list_sorted = sorted(int_node_list)
    str_node_list_sorted = [str(n) for n in int_node_list_sorted]
    return str_node_list_sorted

def property_matrix_nodes_degree(Graph):
    """
    使用networkx计算单层（目的是节点不对齐的双层网络）而写
    # 得到每层节点与度的属性矩阵并以两个向量（数组list形式）返回，这里主要是单层的了
    [('1', 2), ('2', 77), ('3', 4), ('6', 1), ('7', 6), ('8', 2)]
    :param Graph: 双层网络对象
    :param N: 每层的节点N
    :return: 节点没有顺序的度列表（也就是节点并不是从小到大，而是随机的）
    """
    node_sorted = get_sorted_node_byGraph(Graph)
    node_degree = nx.degree(Graph, node_sorted)
    matrix = list(dict(node_degree).values())
    # matrix_0 = get_degrees_by_layer(Graph, Graph.num_nodes_in_layers[0], 0)
    # matrix_1 = get_degrees_by_layer(Graph, Graph.num_nodes_in_layers[1], 1)
    return matrix


def property_matrix_nodes_CC(Graph):
    """
    得到每层节点与聚类系数CC的属性矩阵并以两个向量（数组list形式）返回
    :param Graph:
    :param N:
    :return: 节点没有顺序的聚类系数列表（也就是节点并不是从小到大，而是随机的）
    """
    node_sorted = get_sorted_node_byGraph(Graph)
    CC = list(nx.clustering(Graph, node_sorted).values())
    # CC_1 = CC[:N]
    # CC_2 = CC[N:]
    return CC


def property_matrix_edges_existence_by_node_aligned(Graph, N):
    """
    得到节点对齐的网络中的每层全部边是否存在的属性矩阵并以两个向量（数组list形式）返回
    :param Graph:
    :param N:
    :return:
    """
    edges_existence_1 = {}
    # edges_existence_2 = {}
    for u in range(N):
        for v in range(u + 1, N):
            if Graph.has_edge(u, v):
                edges_existence_1[(u, v)] = 1
            else:
                edges_existence_1[(u, v)] = 0
            # if Graph.has_edge(u + N, v + N):
            #     edges_existence_2[(u, v)] = 1
            # else:
            #     edges_existence_2[(u, v)] = 0
    return edges_existence_1


def property_matrix_edges_existence_by_node_no_aligned(Graph):
    """
    得到广义（节点不对齐）的双层网络每层全部边是否存在的属性矩阵并以两个向量（数组list形式）返回
    :param Graph:
    :param N:
    :return:
    """
    flag_edges = []
    edges_existence_1 = {}
    nodes = list(Graph.nodes)
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            if (u, v) in flag_edges or (v, u) in flag_edges:
                continue
            if Graph.has_edge(u, v):
                edges_existence_1[(u, v)] = 1
            else:
                edges_existence_1[(u, v)] = 0
            flag_edges.append((u, v))
    return edges_existence_1


def property_matrix_triangles_existence(Graph, N):
    """
    得到每层全部三角形模体是否存在的属性矩阵并以两个向量（数组list形式）返回
    :param Graph:
    :param N:
    :return:
    """
    triangle_existence_1 = {}
    # triangle_existence_2 = {}
    for u in range(N):
        for v in range(u + 1, N):
            for k in range(v + 1, N):
                if Graph.has_edge(u, v) and Graph.has_edge(u, k) and Graph.has_edge(v, k):
                    triangle_existence_1[(u, v, k)] = 1
                else:
                    triangle_existence_1[(u, v, k)] = 0
                # if Graph.has_edge(u + N, v + N) and Graph.has_edge(u + N, k + N) and Graph.has_edge(v + N, k + N):
                #     triangle_existence_2[(u, v, k)] = 1
                # else:
                #     triangle_existence_2[(u, v, k)] = 0
    return triangle_existence_1


"""
依赖于研究学者对网络中某一感兴趣的特定结构进行比较，采用Jaccard相似系数方法
"""


def similarity_binary_property__fun(dict_l1, dict_l2, normalized_fun='jaccard'):
    """
    根据边与三角形结构来求层之间的相似性
    :param P_l1: 层二元字典，需要转化为矩阵
    :param P_l2: 层二元字典，需要转化为矩阵
    :param normalized_fun: 使用归一化函数时要和具体的网络情况
    默认归一化函数使用'jaccard', 其他函数分别是'smc', 'hamann', 'kulczynski'
    SMC 和 Hamann 仅适用于非稀疏、非退化的情况，如果属性向量稀疏，Russel-Rao 也会受到影响
    jaccard 是比较实用的
    :return:
    """
    # 此方法为结构型的属性矩阵，比如边与三角形模体是否存在
    # 首先求 a b c d 四个值
    # a = l1 和 l2 共有的属性数
    # b = l1 具有和 l2 缺少的属性数量
    # c = l1 缺少和 l2 具有的属性数量
    # d = l1 和 l2 都缺少的属性数量

    # 遍历np数组，把其中不匹配的都填0
    # if len(dict_l1) > len(dict_l2):
    #     max_dict = dict_l1
    #     min_dict = dict_l2
    # else:
    #     max_dict = dict_l2
    #     min_dict = dict_l1

    # 这里是根据边结构来改的！
    for key in dict_l1:
        u, v = key
        if dict_l2.get((u, v)) == None and dict_l2.get((v, u)) == None:
            dict_l2[(u, v)] = 0

    for key in dict_l2:
        u, v = key
        if dict_l1.get((u, v)) == None and dict_l1.get((v, u)) == None:
            dict_l1[(u, v)] = 0
    # 对字典的键进行排序
    P_l1 = []
    P_l2 = []
    for key in sorted(dict_l1):
        u, v = key
        if dict_l1.get((u, v)) != None:
            P_l1.append(dict_l1[(u, v)])
        else:
            P_l1.append(dict_l1[(v, u)])
        if dict_l2.get((u, v)) != None:
            P_l2.append(dict_l2[(u, v)])
        else:
            P_l2.append(dict_l2[(v, u)])

    np_p1 = np.array(P_l1)
    np_p2 = np.array(P_l2)

    a = sum(np_p1 * np_p2)
    b = sum(np_p1 * (1 - np_p2))
    c = sum((1 - np_p1) * np_p2)
    d = sum((1 - np_p1) * (1 - np_p2))
    m = a + b + c + d
    if m != len(np_p1) or m != len(np_p2):
        print('传入的两个属性矩阵有错，长度不一致或者其他错误。')
        return
    if normalized_fun == 'jaccard':
        return a / (m - d)
    if normalized_fun == 'smc':
        return (a + d) / m
    if normalized_fun == 'hamann':
        return (a + d - (b + c)) / m
    if normalized_fun == 'kulczynski':
        return a / 2 * (1 / (a + b) + 1 / (a + c))
    if normalized_fun == 'russelrao':
        return a / m


"""
统计聚合函数并使用相对差异比较层相似性
最适合节点度的属性矩阵以及也可以使用聚类系数的属性矩阵，对结构型的属性矩阵判断不明确
"""


# 简单统计汇总函数
def property_matrix_get_zhijing_fun(Graph):
    # zhijing_0 = mx.diameter(Graph.get_layer(0))
    zhijing = nx.diameter(Graph)
    # zhijing_1 = mx.diameter(Graph.get_layer(1))
    return zhijing


def property_matrix_get_avg_val_fun(P_L):
    if P_L is not np.ndarray:
        P_L = np.array(P_L)
    return sum(P_L) / len(P_L)


def property_matrix_get_sd_val_fun(P_L):
    """
    得到每层的标准差
    最适合节点度的属性矩阵以及也可以使用聚类系数的属性矩阵，对结构型的属性矩阵判断不明确
    :param P_L:
    :return:
    """
    if P_L is not np.ndarray:
        P_L = np.array(P_L)
    mean_P_L = property_matrix_get_avg_val_fun(P_L)
    sd_molecular = sum(np.square(P_L - mean_P_L))
    return np.sqrt(sd_molecular / len(P_L))


def property_matrix_get_skew_val_fun(P_L):
    """
    得到每层的偏度
    最适合节点度的属性矩阵以及也可以使用聚类系数的属性矩阵，对结构型的属性矩阵判断不明确
    :param P_L:
    :return:
    """
    if P_L is not np.ndarray:
        P_L = np.array(P_L)
    mean_P_L = property_matrix_get_avg_val_fun(P_L)
    sd_molecular = sum(np.power(P_L - mean_P_L, 3))
    sd_denominator = len(P_L) * np.power(property_matrix_get_sd_val_fun(P_L), 3)
    return sd_molecular / sd_denominator


def property_matrix_get_kurt_val_fun(P_L):
    """
    得到每层的峰度
    最适合节点度的属性矩阵以及也可以使用聚类系数的属性矩阵，对结构型的属性矩阵判断不明确
    :param P_L:
    :return:
    """
    if P_L is not np.ndarray:
        P_L = np.array(P_L)
    mean_P_L = property_matrix_get_avg_val_fun(P_L)
    sd_molecular = sum(np.power(P_L - mean_P_L, 4))
    sd_denominator = len(P_L) * np.power(property_matrix_get_sd_val_fun(P_L), 4)
    return sd_molecular / sd_denominator


def property_matrix_get_jarqueBera_val_fun(P_L):
    """
    得到每层的Jarque-Bera 统计
    最适合节点度的属性矩阵以及也可以使用聚类系数的属性矩阵，对结构型的属性矩阵判断不明确
    :param P_L:
    :return:
    """
    if P_L is not np.ndarray:
        P_L = np.array(P_L)
    return len(P_L) / 6 * (np.square(property_matrix_get_skew_val_fun(P_L)) + 0.25 * np.square(
        property_matrix_get_kurt_val_fun(P_L) - 3))


def property_matrix_relative_difference_fun(fun_val_pl1, fun_val_pl2):
    """
    相对差异来比较统计聚类函数得出其层相似性，当然是越小越好！
    :param fun_val_pl1:
    :param fun_val_pl2:
    :return:
    """
    if fun_val_pl1 + fun_val_pl2 == 0:
        return 0
    return 2 * (abs(fun_val_pl1 - fun_val_pl2)) / (abs(fun_val_pl1) + abs(fun_val_pl2))

"""
    聚类评估方法，纯度purity和F-measure
"""
def purity_single(community_list1, community_list2, G1_nodes, G2_nodes):
    len_community_list1 = len(community_list1)
    len_community_list2 = len(community_list2)
    sum_k = 0
    for i in range(len_community_list1):
        temp_list = community_list1[i]
        if (len(temp_list) < 2) and (temp_list[0] not in G1_nodes):
            continue
        # 检测被匹配的网络的每一个社区与当前匹配社会的共同最大的节点数
        max_cun = 0
        for j in range(0, len_community_list2):
            temp_list2 = community_list2[j]
            if (len(temp_list2) < 2) and (temp_list2[0] not in G2_nodes):
                continue
            cun = get_list_intersection_length(temp_list, community_list2[j])
            if cun > max_cun:
                max_cun = cun
        # print(max_cun)
        sum_k += max_cun

    return sum_k / len(G1_nodes)

def get_list_intersection_length(list1, list2):
    return len(set(list1).intersection(set(list2)))

def get_Graph_nodes(layer_fileName, sep, has_name):
    """
    得到文件中的节点个数
    :param layer_fileName:
    :param sep: 边链接节点的分割符
    :param has_name: 如果有列名则为1，否则为0
    :return:
    """
    nodes = []
    with open(layer_fileName, 'r') as f:
        if has_name:
            next(f)
        for line in f:
            strs = line.strip().split(sep)
            node1 = int(strs[0])
            node2 = int(strs[1])
            if node1 not in nodes:
                nodes.append(node1)
            if node2 not in nodes:
                nodes.append(node2)
    return nodes

def based_community_layer_similarity_by_purity(layer_edge_1, layer_edge_2):
    """
    这里的格式，txt格式，1 2 以空格当做分隔符
    文件不允许有多余的数据，格式为从第一行开始一行一条边
    :param layer_edge_1:
    :param layer_edge_2:
    :return:
    """
    G1_nodes = get_Graph_nodes(layer_edge_1, ' ', 0)
    G2_nodes = get_Graph_nodes(layer_edge_2, ' ', 0)
    Gi_1 = ig.Graph.Read_Edgelist(layer_edge_1)
    Gi_2 = ig.Graph.Read_Edgelist(layer_edge_2)  # 基于这些连边使用igraph创建一个新网络
    G1 = Gi_1.as_undirected()
    G2 = Gi_2.as_undirected()
    # 检测到的社区+
    community_list1 = G1.community_multilevel(weights=None, return_levels=False)
    community_list2 = G2.community_multilevel(weights=None, return_levels=False)
    # print(community_list1)
    # print(community_list2)
    # 检测purity
    purity1 = purity_single(community_list1, community_list2, G1_nodes, G2_nodes)
    purity2 = purity_single(community_list2, community_list1, G2_nodes, G1_nodes)
    if (purity1 + purity2) == 0 :
        print(f'计算purity的纯度0')
        return 0
    purity = 2 * purity1 * purity2 / (purity1 + purity2)
    print(f'计算purity的纯度{purity}')
    return purity

def f_measure(community_list1, community_list2, G1_nodes, G2_nodes):
    len_community_list1 = len(community_list1)
    len_community_list2 = len(community_list2)
    # 计算聚类F-Measure的平均值
    sum_f = 0
    len_community_real = 0
    for i in range(len_community_list1):
        # 计算单个F-Measure
        temp_list = community_list1[i]
        if (len(temp_list) < 2) and (temp_list[0] not in G1_nodes):
            continue
        # 计算 prec预测值
        len_community_real += 1
        max_cun = 0
        for j in range(0, len_community_list2):
            temp_list2 = community_list2[j]
            if (len(temp_list2) < 2) and (temp_list2[0] not in G2_nodes):
                continue
            cun = get_list_intersection_length(temp_list, community_list2[j])
            if cun > max_cun:
                max_cun = cun
        prec_val = max_cun / len(temp_list)

        # 计算 recall 召回值
        # 这个怎么求的呢？是根据G1网络的每一个社区进行遍历，
        # 遍历第一个社区，第一个社区中的节点与G2网络中的每一个社区进行遍历，找到G2社区中与该社区
        # 有共同节点数最大的G2中的社区，得到其社区的节点的总个数值 Mji (总个数值 Mji 是共有节点的个数，G1网络中没有的节点则不算)
        ext_coms = {cid: nodes for cid, nodes in enumerate(community_list2) if len(nodes) > 2 or ((len(nodes) > 2) and(nodes[0] not in G2_nodes))}
        node_to_com = defaultdict(list)
        for cid, nodes in list(ext_coms.items()):
            for n in nodes:
                node_to_com[n].append(cid)

        ids = {}
        for n in temp_list:
            try:
                idd_list = node_to_com[n]
                for idd in idd_list:
                    if idd not in ids:
                        ids[idd] = 1
                    else:
                        ids[idd] += 1
            except KeyError:
                pass
        maximal_match = {label: p for label, p in list(ids.items()) if p == max(ids.values())}
        if maximal_match == {}:
            f_i = 0
        else :
            label = list(maximal_match.keys())[0]
            m_ji = 0
            for n in community_list2[label]:
                if n in G1_nodes:
                    m_ji += 1
            recall_val = max_cun / m_ji
            f_i = 2 * prec_val * recall_val / (prec_val + recall_val)
        # print('单个f值为: ' , f_i)
        # print(f'prec{max_cun}/{len(temp_list)}recall{max_cun}/{m_ji}')
        sum_f += f_i
    print(f'len_community_real: {len_community_real}')
    # print(len_community_real)
    return sum_f / len_community_real

def based_community_layer_similarity_by_f_measure(layer_edge_1, layer_edge_2):

    G1_nodes = get_Graph_nodes(layer_edge_1,' ',0)
    G2_nodes = get_Graph_nodes(layer_edge_2,' ',0)
    Gi_1 = ig.Graph.Read_Edgelist(layer_edge_1)
    Gi_2 = ig.Graph.Read_Edgelist(layer_edge_2)  # 基于这些连边使用igraph创建一个新网络
    G1 = Gi_1.as_undirected()
    G2 = Gi_2.as_undirected()
    # 检测到的社区
    community_list1 = G1.community_multilevel(weights=None, return_levels=False)
    community_list2 = G2.community_multilevel(weights=None, return_levels=False)

    # 检测purity
    f_measure_1 = f_measure(community_list1, community_list2, G1_nodes, G2_nodes)
    f_measure_2 = f_measure(community_list2, community_list1, G2_nodes, G1_nodes)
    if f_measure_1 == 1:
        print(f'f_measure_1的计算值为1')
        return 0
    if f_measure_2 == 1:
        print(f'f_measure_2的计算值为1')
        return 0
    if (f_measure_1 + f_measure_2) == 0:
        print(f'计算f_measure的纯度0')
        return 0
    f_val = 2 * f_measure_1 * f_measure_2 / (f_measure_1 + f_measure_2)
    print(f'计算f_measure的纯度{f_val}')
    return f_val


def D_value(a_addr, b_addr, filename_single, filename_multi, is_ling_modle=False, W1=0.45, W2=0.45, W3=0.1):
    """
    计算a_addr, b_addr 两个图的相似性，D结果值为0就是相似的
    :param W1: 设置好的权重值0.45
    :param W2: 设置好的权重值0.45
    :param W3: 设置好的权重值0.1
    :param a_addr: 图1 下层
    :param b_addr: 图2 上层
    :return: 返回相似度，为0则为相似的，取值在0-1之间
    """
    # 生成网络图
    name_list_a, name_list_b = ['city_name', 'city_id_name'], ['city_name', 'city_id_name']
    # name_list_a, name_list_b = [], []

    total_city, G_a, G_b = read_csv(a_addr, b_addr,is_ling_modle, name_list_a, name_list_b, filename_single, filename_multi)

    N = len(total_city)
    nodes = range(N)  # 节点列表
    first_value = first(G_a, G_b)
    second_value = second(G_a, G_b, N)
    third_value = third(G_a, G_b, nodes)
    print("first_value", first_value)
    print("second_value", second_value)
    print("third_value", third_value)
    return W1 * first_value + W2 * second_value + W3 * third_value * 0.5

"""
    信息度量IC计算开始
"""
def information_content_IC(AM, directed=0):
    """
    该统计量的使用必须两层网络的节点数相同
    AM: adjacency matrix numpy类型
    directed：1 if the network is directed,
              0 otherwise
    Outputs:
        Info: the IC of the network
        NormInfo: the normalized IC
    """
    numNodes = AM.shape[0]  # 节点个数
    linkDensity = sum(sum(AM)) / (numNodes * numNodes)  # 节点个数
    Info = 0

    # 计算边的HD
    distanceMatrix = np.arange(numNodes * numNodes).reshape(numNodes, numNodes)
    for n1 in range(numNodes):
        for n2 in range(numNodes):
            if n1 == n2:
                distanceMatrix[n1][n2] = numNodes
                continue
            distanceMatrix[n1][n2] = HD(AM[n1, :], AM[n2, :], AM[:, n1], AM[:, n2])  # TODO

    # 计算每一个节点所损失的信息
    for k in range(numNodes - 1):
        tInfo, AM, distanceMatrix = oneIteration(AM, distanceMatrix)
        #         print(Info)
        Info += tInfo

    # 不知道还用不用标准化输出
    # NormInfo = Info / Normalize(linkDensity, numNodes, directed)
    # print(f'计算出的信息度量为：{Info}'), NormInfo

    return Info

# 一次迭代
def oneIteration(AM, distanceMatrix):
    # print(f"AM={AM}")
    numNodes = AM.shape[0]  # 节点个数
    minDist = np.min(np.min(distanceMatrix))  # 距离矩阵的最小值
    maxDist = np.max(np.max(distanceMatrix))  # 距离矩阵的最大值

    if minDist < (2 * numNodes - maxDist):
        aX = np.min(distanceMatrix, axis=0)
        bX = np.argmin(distanceMatrix, axis=0)

        bestHD = np.min(aX, axis=0)
        cX = np.argmin(aX, axis=0)
    else:
        aX = np.max(distanceMatrix, axis=0)
        bX = np.argmax(distanceMatrix, axis=0)

        bestHD = np.max(aX, axis=0)
        cX = np.argmax(aX, axis=0)

    # 最好的节点和aX中最好的节点标号
    bestNode = bX[cX]
    bestNode2 = cX

    if bestNode == bestNode2:
        bestNode = 0
        bestNode2 = 0

    p = bestHD / (numNodes * 2)
    Info = 0
    if p > 0 and p < 1:
        Info = - p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
        Info = Info * numNodes * 2

    newDM1 = distanceMatrix

    for n1 in range(numNodes):
        for n2 in range(numNodes):
            if n1 == n2:
                continue

            newDM1[n1][n2] = newDM1[n1][n2] + np.double(AM[n1][bestNode] == AM[n2][bestNode]) - 1
            newDM1[n1][n2] = newDM1[n1][n2] + np.double(AM[bestNode][n1] == AM[bestNode][n2]) - 1
    # print(f"bestNode={bestNode}-----numNodes={numNodes}")
    offset1 = [i for i in range(bestNode - 1)] + [i for i in range(bestNode + 1, numNodes)]
    # print(f"offset1={offset1}")
    offset1_len = len(offset1)
    newDM1 = newDM1[offset1[0]:offset1[offset1_len - 1] + 1, offset1[0]:offset1[offset1_len - 1] + 1]

    newDM2 = distanceMatrix

    for n1 in range(numNodes):
        for n2 in range(numNodes):
            if n1 == n2:
                continue

            newDM2[n1][n2] = newDM2[n1][n2] + np.double(AM[n1][bestNode2] == AM[n2][bestNode2]) - 1
            newDM2[n1][n2] = newDM2[n1][n2] + np.double(AM[bestNode2][n1] == AM[bestNode2][n2]) - 1

    offset2 = [i for i in range(bestNode2 - 1)] + [i for i in range(bestNode2 + 1, numNodes)]
    # print(f"offset2={offset2}")
    offset2_len = len(offset2)
    newDM2 = newDM2[offset2[0]:offset2[offset2_len - 1] + 1, offset2[0]:offset2[offset2_len - 1] + 1]

    maxV1 = np.max(np.max(newDM1))
    maxV2 = np.max(np.max(newDM2))
    minV1 = np.min(np.min(newDM1))
    minV2 = np.min(np.min(newDM2))

    if maxV1 < (2 * (numNodes - 1) - minV1):
        maxV1 = 2 * (numNodes - 1) - minV1

    if maxV2 < (2 * (numNodes - 1) - minV2):
        maxV2 = 2 * (numNodes - 1) - minV2

    if maxV1 > maxV2:
        newAM = AM[offset1[0]:offset1[offset1_len - 1] + 1, offset1[0]:offset1[offset1_len - 1] + 1]
        newDM = newDM1
    else:
        newAM = AM[offset2[0]:offset2[offset2_len - 1] + 1, offset2[0]:offset2[offset2_len - 1] + 1]
        newDM = newDM2
    return Info, newAM, newDM

def HD(vector1, vector2, vector3, vector4):
    sizeVector = np.max(vector1.shape)
    answ = np.sum(np.double(vector1 == vector2)) + np.sum(np.double(vector3 == vector4))

    answ = 2 * sizeVector - answ
    return answ

def approxF(p):
    if p == 0 or p == 1:
        approxInfo = 0
        return approxInfo

    approxInfo = - p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return approxInfo

def Normalize(p, numNodes, directed):
    if directed == 1:
        KMax = 0.1625 - 1.822 * np.power((numNodes + 10.08), -1.098)
    else:
        KMax = 0.1622 - 0.6298 * np.power((numNodes + 5.905), -1.132)

    alpha = KMax * 4
    effP = alpha * p - alpha * p * p

    Info = 0

    for k in range(numNodes, 1, -1):
        temp = (2 * k) * approxF(effP)
        Info = Info + temp

    return Info

"""
    信息度量IC计算结束
"""

"""
    第三项计算开始
"""

def third(Graph_a, Graph_b, nodes):
    # FIRST
    p_a = P_a(Graph_a, nodes)
    p_b = P_a(Graph_b, nodes)
    first_value = J_P_a(p_a, p_b)

    # SECOND
    a = G_C(Graph_a)
    b = G_C(Graph_b)
    p_c_a = P_a(a, nodes)
    p_c_b = P_a(b, nodes)
    second_value = J_P_a(p_c_a, p_c_b)
    third_value = first_value + second_value
    return third_value


def G_C(Graph):
    # 补图
    GC = nx.complement(Graph)  # 补图
    return GC


def J_P_a(p_a, p_b):
    # 计算JS散度
    # jS_divergence = JS_divergence(p_a, p_b)
    jS_divergence = D_JS(p_a, p_b)
    JS_divergence_formula = math.sqrt(jS_divergence / math.log(2))
    return JS_divergence_formula


def D_JS(P1, P2):
    P3 = np.sum([P1, P2], axis=0)
    P5 = list(P3 * 0.5)
    n = 0.5 * D_KL(P1, P5)
    m = 0.5 * D_KL(P2, P5)
    js_value = n + m
    return js_value


def D_KL(p, q):
    sum = 0
    for i, j in zip(p, q):
        m = i / j
        if (m > 0):
            n = i * math.log(m)
        sum = sum + n
    return sum


def P_a(Graph, nodes):
    # # 拉姆达最大值分之一
    N = len(nodes)
    n = 1 / N
    # 计算katz中心性 基于其邻居的中心性计算节点的中心性 TODO
    centrality = nx.katz_centrality_numpy(Graph, n)
    c = list(centrality.values())
    c.sort()
    V = np.sum(c)
    # print(V)

    # 求α中心性
    p = []  # p_each
    for v_i in c:
        p_each = v_i + V / N
        # print("p_each", p_each)
        p.append(p_each)
    return p


"""
    第三项计算结束
"""

"""
    第二项计算开始
"""


def second(Graph_a, Graph_b, N):
    nnd_sub = math.sqrt(NND(Graph_a, N)) - math.sqrt(NND(Graph_b, N))
    second_value = abs(nnd_sub)
    return second_value


def NND(Graph, N):
    # d = G_D(G)  # 网络的直径
    d = nx.diameter(Graph)
    j_p = J_P(Graph, N)  # JS散度
    nnd = j_p / math.log(d + 1)
    print("NND", nnd)
    return nnd


def J_P(G, N):
    # UG_J
    # j_value_max,p_i_j
    j_value_max, p_i_j = P_I_J(G)
    miu_j = UG(G)
    sum = 0
    for i, v in p_i_j.items():
        for j, i_v in v.items():
            m = i_v / miu_j[j]
            i_j = i_v * math.log(m)
            sum += i_j

    j_p_value = sum / N
    return j_p_value


"""
    第二项计算结束
"""

"""
    第一项计算开始
"""


def first(Graph_a, Graph_b):
    u_g_a = UG(Graph_a)
    u_g_b = UG(Graph_b)

    a_len = len(u_g_a)
    b_len = len(u_g_b)
    max_v = max(a_len, b_len)

    # 将u_g_au_g_b与参数不足之处补0
    if (a_len != max_v):
        adj = b_len - a_len
        for adi_index in range(0, adj):
            # miu_j_a.append(0)
            u_g_a[a_len + 1] = 0
            a_len += 1
    elif (b_len != max_v):
        adj = a_len - b_len
        for adi_index in range(0, adj):
            u_g_b[b_len + 1] = 0
            b_len += 1

    miu_j_a_list = list(u_g_a.values())
    miu_j_b_list = list(u_g_b.values())

    """JS散度"""
    d_js_value = JS_divergence(miu_j_a_list, miu_j_b_list)
    first = math.sqrt(d_js_value / math.log(2))
    return first.real


def JS_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    M = (p + q) * 0.5
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def UG(Graph):
    N = len(Graph.nodes)
    j_value_max, p_i_j = P_I_J(Graph)
    miu_j_list = {}
    for j in range(1, j_value_max + 1):
        miu_j = P_I(p_i_j, j, N)
        # print(miu_j)
        miu_j_list[j] = miu_j

    # print("miu_j_list",miu_j_list)
    return miu_j_list


def P_I(p_i_j, j, N):
    miu_j_each = []
    for i in p_i_j:
        # 判断内部的key是否存在j值 has_key
        single = j in p_i_j[i].keys()
        if (single):
            pass
        else:
            p_i_j[i][j] = 0
        miu_j_each.append(p_i_j[i][j])
    miu_j = sum(miu_j_each) / N
    return miu_j


def P_I_J(Graph):
    N = len(Graph.nodes)
    p_i_j = {}
    # n_i_j_1 = nx.single_source_bellman_ford_path_length(G, 1)
    n_i_j = N_I_J(Graph)
    # p_i_j=n_i_j/(N-1)
    j_value_max = 0  # j的最大值
    for i, v in n_i_j.items():
        # print("i",i,"v",v)
        p_j = {}
        j_value = []
        for j, value in v.items():
            # print("j",j,"value",value)
            value = value / (N - 1)
            p_j[j] = value
            j_value.append(j)  # 每个图的j的值
        p_i_j[i] = p_j
        j_value_max = max(j_value) if j_value_max < max(j_value) else j_value_max
    # j_value_max = max(j_value_max)
    # print("j_value_max",j_value_max)
    # print("p_i_j", p_i_j)
    return j_value_max, p_i_j


def N_I_J(Graph):
    sub_node_list = []
    # 得到其联通组件，得到所有的联通子图
    for c in nx.connected_components(Graph):
        subgraph = Graph.subgraph(c)
        a = []
        if (len(subgraph) > 1):
            for pp in subgraph:
                a.append(pp)
            # nx.draw_networkx(subgraph)
            sub_node_list.append(a)

    # 虽然是有向图，但是求无向图的最短路径来算
    # 也就是求每个联通子图的其中每个节点到其他可到达的所有节点的最短路径
    n_i_list = {}  # 学姐这里其实是有问题的 TODO
    for sub_node in sub_node_list:
        mm = sub_node.copy()
        for node in sub_node:
            n_i_list[node] = [nx.shortest_path_length(Graph, target=node, source=target) for target in mm if
                              target != node]

    #
    n_i_j_list = {}
    for k, v in n_i_list.items():
        a = {}
        max_v = max(v)
        for v_index in range(1, max_v + 1):
            a[v_index] = v.count(v_index)
        n_i_j_list[k + 1] = a
    return n_i_j_list


def read_csv(a_addr, b_addr, is_ling_modle, name_list_a, name_list_b, filename_single, filename_multi):
    # 读取数据
    df_a = pd.read_csv(a_addr)
    df_b = pd.read_csv(b_addr)
    # 获取单层网络
    get_single_cities_num(df_a, name_list_a, filename_single + '_a')
    get_single_cities_num(df_b, name_list_b, filename_single + '_b')
    # 获取双层网络
    get_multi_cities_num(df_a, df_b, name_list_a, name_list_b, filename_multi)  # 将不同网络的城市结点写入文件 创建

    # 读取两个网络中的总节点, 此时节点还是汉字的形式
    total_nodes = read_total_nodes(filename_multi)
    weighted_net_a_lines = links_weighted(df_a)
    weighted_net_b_lines = links_weighted(df_b)

    # net_totals_array = create_adjacency_matrix(total_nodes)  # 邻接矩阵

    # 将城市名依依替换为数字名，返回节点列表和边列表以及对应的权重值
    nodes_list_all_01, edges_list_weight_01 = net_nodes_to_number(total_nodes, weighted_net_a_lines)
    nodes_list_all_02, edges_list_weight_02 = net_nodes_to_number(total_nodes, weighted_net_b_lines)

    net_weighted_array_01 = create_network_graph(nodes_list_all_01, edges_list_weight_01)
    net_weighted_array_02 = create_network_graph(nodes_list_all_02, edges_list_weight_02)
    # 富节点查看节点个数取百分之二十左右 大于14的（这个仅针对于衡量零模型与原始网络的）
    # degrees = [d for n,d in net_weighted_array_02.degree()]
    #
    # for degree in range(74):
    #     tmp_degrees = [d for d in degrees if d>degree]
    #     if (len(tmp_degrees) / 300) <= 0.2:
    #         rich_de = degree
    #         print(len(tmp_degrees) / 300)
    #         break

    # 对02进行随机零模型转换
    if is_ling_modle:
        # 先测试单层层内的
        net_weighted_array_02 = unm.rich_club_break(net_weighted_array_02, k=14, nswap=20000, max_tries=100000000)

    return total_nodes, net_weighted_array_01, net_weighted_array_02


def create_network_graph(cities_nodes, weighted_link):
    """
    搞网络
    :param cities_nodes:
    :param weighted_link:
    :return:
    """
    net = nx.Graph()
    net.add_nodes_from(cities_nodes)
    net.add_weighted_edges_from(weighted_link)
    return net


def net_nodes_to_number(total_nodes, weighted_net_lines):
    # 从总的节点的列表中的文字转换成数字
    nodes_map = {}
    nodes_list_old = []
    index = 0
    for node in total_nodes:
        if node not in nodes_list_old:
            nodes_list_old.append(node)
            nodes_map[node] = index
            index += 1  # 起始节点从0开始

    nodes_list_all = [k for i, k in nodes_map.items()]

    # 从边的对应关系中找到用数字代表的关系
    edges_list_weight = [(nodes_map[node1], nodes_map[node2], weight) for node1, node2, weight in weighted_net_lines]
    return nodes_list_all, edges_list_weight


def create_adjacency_matrix(total_nodes):
    """建立邻接矩阵"""
    array = pd.DataFrame(np.zeros((len(total_nodes), len(total_nodes))),
                         index=total_nodes, columns=total_nodes)
    return array


def links_weighted(df):
    """
    把df转换为元祖列表
    :param df:
    :return:
    """
    weighted_lines = [tuple(i) for i in df.values[:, :3]]
    return weighted_lines


def read_total_nodes(filename):
    nodeslist = []
    with open('./data/multi_nodes_list' + filename + '.txt', 'r', encoding="utf_8_sig") as file:
        nodes_data = file.readlines()
        for i in nodes_data:
            i = i.strip("\n")
            nodeslist.append(i)
    return nodeslist


def get_multi_cities_num(df_a, df_b, name_list_a, name_list_b, filename):
    """
    :param df_a: 网络A的df数据 下层
    :param df_b: 网络B的df数据 上层
    :param name_list_a: 文件A里的属性节点列表
    :param name_list_b: 文件B里的属性节点列表
    :param filename: 给新文件的命名
    :return:
    """
    """获取双层网络总城市结点数"""
    total_nodes_a = get_df_namelist_nodes(df_a, name_list_a)
    total_nodes_b = get_df_namelist_nodes(df_b, name_list_b)
    set_total_nodes_ab = set(total_nodes_a + total_nodes_b)

    with open('./data/multi_nodes_list' + filename + '.txt', 'w', encoding="utf_8_sig") as file:
        for i in set_total_nodes_ab:
            file.write(i)
            file.write("\n")


def get_single_cities_num(df, name_list, filename):
    """
    :param df: 网络的df数据 下层
    :param name_list: 文件里的属性节点列表
    :param filename: 给新文件的命名
    :return:
    """
    # 将每个网络的节点写入到文件
    # 将DataFrame下的所有节点都存入
    total_nodes = get_df_namelist_nodes(df, name_list)
    set_total_nodes = set(total_nodes)

    with open('./data/single_nodes_list' + filename + '.txt', 'w', encoding="utf_8_sig") as file:
        for i in set_total_nodes:
            file.write(i)
            file.write("\n")


def get_df_namelist_nodes(df, name_list):
    total_nodes = []
    for name in name_list:
        total_nodes += df[name].tolist()
    return total_nodes


"""
    第一项计算结束
"""

"""
互惠性 r指标
"""


def reciprocity_get_data(filename, to_filename, year, month, day):
    """
    从总数据中截取某一天，或某一月   这个数据的参数属性到时候看数据来改
    :param filename:
    :param year:
    :param month:
    :param day:
    :return:
    """
    car = pd.read_csv(filename)
    arr = []
    dep = []
    number = []
    year = []
    month = []
    day = []
    if day:
        a = car[(car.year == year) & (car.month == month) & (car.day == day)].index.tolist()
    else:
        a = car[(car.year == year) & (car.month == month)].index.tolist()
    for i in a:
        arr.append(car['arr_city'][i])
        dep.append(car['dep_city'][i])
        number.append(car['number'][i])
        year.append(car['year'][i])
        month.append(car['month'][i])
        day.append(car['day'][i])

    new_car = pd.DataFrame()
    new_car['arr_city'] = arr
    new_car['dep_city'] = dep
    new_car['number'] = number
    new_car['year'] = year
    new_car['month'] = month
    new_car['day'] = day

    new_car.to_csv('to_filename' + '.csv', index=False)


def reciprocity_city_name(filename):
    # author = pd.read_csv('new_car_2017_all.csv')
    city = pd.read_csv(filename)
    text = []
    car_name = []
    car_num = []

    for i in range(len(city)):
        text.append(city['arr_city'][i])
        text.append(city['dep_city'][i])

    car_name = list(set(text))
    author_article_num = pd.DataFrame()
    author_article_num['city'] = car_name
    author_article_num.to_csv('car_city_name.csv', index=False)


def reciprocity_get_month_r(data_filename, car_name_filename):
    """
    计算某一个月中每一天各城市的与互惠性相关的值，例如对称强度，非对称强度，
    最后会生成四个文件，分别对应对称强度，非对称强度，入强度，出强度
    :return:
    """
    # car = pd.read_csv('new_car_2017-11_all.csv')
    # car_name = pd.read_csv('car_city_name.csv')
    car = pd.read_csv(data_filename)
    car_name = pd.read_csv(car_name_filename)

    car1 = car.groupby(car['day'])
    city_two_num = pd.DataFrame()
    city_one_num = pd.DataFrame()
    city_in_num = pd.DataFrame()
    city_out_num = pd.DataFrame()

    for i in range(len(car_name)):
        s_in = []
        s_out = []
        city_name = car_name['city'][i]
        b = car[(car.arr_city == city_name)].index.tolist()
        c = car[(car.dep_city == city_name)].index.tolist()
        arr_city = []
        dep_city = []
        number = []
        day = []
        car_city = pd.DataFrame()
        for j in b:
            arr_city.append(car['arr_city'][j])
            dep_city.append(car['dep_city'][j])
            number.append(car['number'][j])
            day.append(car['day'][j])

        for j in c:
            arr_city.append(car['arr_city'][j])
            dep_city.append(car['dep_city'][j])
            number.append(car['number'][j])
            day.append(car['day'][j])

        car_city['arr_city'] = arr_city
        car_city['dep_city'] = dep_city
        car_city['number'] = number
        car_city['day'] = day

        car_city_new = car_city.groupby(car_city['day'])
        if len(car_city_new) >= 30:
            in_two_all = []
            in_num_all = []
            out_num_all = []
            in_one_all = []
            for day, car_city_day in car_city_new:
                in_two = 0
                in_num = 0
                out_num = 0
                in_one = 0
                car_city_day.index = range(len(car_city_day))
                print(car_city_day)

                ####入强度
                d = car_city_day[
                    (car_city_day.arr_city == city_name)].index.tolist()
                if len(d) > 0:
                    for z in d:
                        in_num += car_city_day['number'][z]
                in_num_all.append(in_num)

                ####出强度
                f = car_city_day[
                    (car_city_day.dep_city == city_name)].index.tolist()
                if len(f) > 0:
                    for z in f:
                        out_num += car_city_day['number'][z]
                out_num_all.append(out_num)

                ####对称强度
                for i in range(len(car_city_day)):
                    e = car_city_day[
                        (car_city_day.arr_city == car_city_day['dep_city'][i]) & (
                                car_city_day.dep_city == car_city_day['arr_city'][i])].index.tolist()
                    if len(e) > 0:
                        number = min(car_city_day['number'][i], car_city_day['number'][e[0]])
                        in_two += number / 2
                in_two_all.append(in_two)
                print(in_num)
                print(in_two)

                ####非对称强度
                # for i in range(len(car_city_day)):
                #     g= car[
                #         (car.arr_city == car['dep_city'][i]) & (car.dep_city == car['arr_city'][i])].index.tolist()
                #     if len(g) > 0:
                #         number = min(car['number'][i], car['number'][g[0]])
                #         in_one = in_one - 2 * number
                in_one_all.append(in_num + out_num - 2 * in_two)

            city_two_num[city_name] = in_two_all
            city_one_num[city_name] = in_one_all
            city_in_num[city_name] = in_num_all
            city_out_num[city_name] = out_num_all

    city_two_num.to_csv('car_city_2017-11_in_two.csv', index=False)
    city_one_num.to_csv('car_city_2017-11_in_one.csv', index=False)
    city_in_num.to_csv('car_city_2017-11_in_num.csv', index=False)
    city_out_num.to_csv('car_city_2017-11_out_num.csv', index=False)


def reciprocity_get_r_day(filename):
    """
    计算某一天交通网络的互惠性
    :return:
    """
    car = pd.read_csv(filename)
    W = sum(list(car['number'].values))
    double_arr = []
    double_dep = []
    double_number = []

    for i in range(len(car)):
        a = car[(car.arr_city == car['dep_city'][i]) & (car.dep_city == car['arr_city'][i])].index.tolist()
        if len(a) > 0:
            double_arr.append(car['arr_city'][i])
            double_arr.append(car['arr_city'][a[0]])

            double_dep.append(car['dep_city'][i])
            double_dep.append(car['dep_city'][a[0]])

            double_number.append(car['number'][i])
            double_number.append(car['number'][a[0]])

    double_car = pd.DataFrame()

    double_car['arr_city'] = double_arr
    double_car['dep_city'] = double_dep
    double_car['number'] = double_number
    double_car = double_car.drop_duplicates(subset=None, keep='first', inplace=False)
    double_car.index = range(len(double_car))
    print(double_car)
    double_car.to_csv('double_car_2017-08-06.csv', index=False)
    print(len(double_car))

    weight_double = 0

    for i in range(len(double_car)):
        weight = 0
        a = double_car[(double_car.arr_city == double_car['dep_city'][i]) & (
                double_car.dep_city == double_car['arr_city'][i])].index.tolist()
        if len(a) > 0:
            print(double_car['number'][i], double_car['number'][a[0]])
            weight = min(double_car['number'][i], double_car['number'][a[0]])
            weight_double += weight

    print(weight_double)
    print(weight_double / W)


def reciprocity_get_r_month(fielname, to_filename):
    """
    计算某一个月每一天网络的互惠性，也就是说会得到30个或31个互惠性值
    :param fielname:
    :return:
    """
    car = pd.read_csv(fielname)
    list = np.array(range(32))
    list = list[1:32]
    r = []
    for i in list:
        a = car[(car.day == i)].index.tolist()
        print(a)

        ###根据天进行划分，分别计算
        car_new = pd.DataFrame()
        arr_city = []
        dep_city = []
        number = []
        for j in a:
            arr_city.append(car['arr_city'][j])
            dep_city.append(car['dep_city'][j])
            number.append(car['number'][j])
        # random.shuffle(number)
        car_new['arr_city'] = arr_city
        car_new['dep_city'] = dep_city
        car_new['number'] = number
        print(car_new)

        ###计算r
        W = car_new['number'].values.sum()
        print(W)
        weight_double = 0

        for i in range(len(car_new)):
            weight = 0
            a = car_new[(car_new.arr_city == car_new['dep_city'][i]) & (
                    car_new.dep_city == car_new['arr_city'][i])].index.tolist()
            if len(a) > 0:
                weight = min(car_new['number'][i], car_new['number'][a[0]])
                weight_double += weight

        print(weight_double)
        r.append(weight_double / W)

    car_r = pd.DataFrame()

    car_r['day'] = list
    car_r['r_num'] = r

    print(car_r)
    car_r.to_csv(to_filename, index=False)


def reciprocity_get_citys_day(filenames):
    """
    计算某一天各城市的相关指标
    :param filenames:
    :return:
    """

    # car = pd.read_csv('new_car_2017-08-06_all.csv')
    # car_num = pd.read_csv('car_num_2017-08-06.csv')
    # double_car = pd.read_csv('double_car_2017-08-06.csv')
    car = pd.read_csv(filenames[0])
    car_num = pd.read_csv(filenames[1])
    double_car = pd.read_csv(filenames[2])

    car_num['in'] = 0
    car_num['out'] = 0
    car_num['in+out_one'] = 0
    car_num['in+out_two'] = 0

    print(car_num)

    ##计算对称强度
    for i in range(len(double_car)):
        a = car_num[(car_num.car_name == double_car['arr_city'][i])].index.tolist()
        b = double_car[
            (double_car.arr_city == double_car['dep_city'][i]) & (
                    double_car.dep_city == double_car['arr_city'][i])].index.tolist()
        if len(b) > 0:
            number = min(double_car['number'][i], double_car['number'][b[0]])
            car_num['in+out_two'][a[0]] += number / 2

    ###计算非对称强度
    for i in range(len(car_num)):
        car_num['in+out_one'][i] = car_num['num'][i]

    print(car_num)

    for i in range(len(car)):
        a = car_num[(car_num.car_name == car['arr_city'][i])].index.tolist()
        b = car[
            (car.arr_city == car['dep_city'][i]) & (car.dep_city == car['arr_city'][i])].index.tolist()
        if len(b) > 0:
            number = min(car['number'][i], car['number'][b[0]])
            car_num['in+out_one'][a[0]] = car_num['in+out_one'][a[0]] - 2 * number

    print(car_num)

    ###计算出入度
    for i in range(len(car)):
        a = car_num[(car_num.car_name == car['arr_city'][i])].index.tolist()
        b = car_num[(car_num.car_name == car['dep_city'][i])].index.tolist()
        if len(b) > 0 and len(a) > 0:
            car_num['in'][a[0]] += car['number'][i]
            car_num['out'][b[0]] += car['number'][i]

    car_num.to_csv('car_num_2017-08-06.csv', index=False)


def reciprocity_get_node(filename):
    """
    节点获取
    :param filename:
    :return:
    """
    # car = pd.read_csv('new_car_2017-08-06_all.csv')
    car = pd.read_csv(filename)
    car.index = range(len(car))
    print(car)

    text = []
    car_name = []
    car_num = []

    for i in range(len(car)):
        text.append(car['arr_city'][i])
        text.append(car['dep_city'][i])

    for i in text:
        if i not in car_name:
            car_name.append(i)

    print(car_name)

    car_go_num = pd.DataFrame()

    car_go_num['car_name'] = car_name
    car_go_num['num'] = 0

    print(car_go_num)

    for i in range(len(car)):
        a = car_go_num[(car_go_num.car_name == car['arr_city'][i])].index.tolist()
        b = car_go_num[(car_go_num.car_name == car['dep_city'][i])].index.tolist()
        car_go_num['num'][a[0]] += car['number'][i]
        car_go_num['num'][b[0]] += car['number'][i]

    print(car_go_num)

    car_go_num.to_csv('car_num_2017-08-06.csv', index=False)
