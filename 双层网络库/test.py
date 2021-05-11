# -*- coding: utf-8 -*-
"""
Created on Apr 19  2021
revised on
@author:  xiaozhi
"""

import ling_model as lm
import networkx as nx
import pandas as pd
import math

# 对两个网络，如BA和Ws网络，也就是生成的双层网络，求进行富人俱乐部效应前后的层间富人俱乐部系数以及生成的新富人俱乐部性质的零模型网络
def inter_richClubTwo_startAndEnd(filename1, filename2, file1_m, file2_n, rich=True):
    '''
    对两个网络，如BA和Ws网络，求进行富人俱乐部效应 前后 的层间富人俱乐部系数
    :param filename1: 要进行节点置换的网络文件在前，文件格式必须是 .csv 格式，节点列名对应 node1, node2
    :param filename2: 不进行节点置换的网络文件，文件格式必须是 .csv 格式，节点列名对应 node1, node2
    :param file1_m: 进行置换网络的可以属于富节点的度值
    :param file2_n: 不进行置换网络的可以属于富节点的度值
    :param rich: 要具有富人俱乐部特性的为True，要分析不具有富人俱乐部特性的为False
    :return: 无返回值
    '''
    dfBA = pd.read_csv(filename1)
    dfws = pd.read_csv(filename2)

    G = lm.create_MultilayerGraphFromPandasEdgeList(dfBA, dfws, source_df1='node1', target_df1='node2', source_df2='node1',
                                                 target_df2='node2')

    N = int(len(G.nodes) / 2)
    print(f'未生成富人俱乐部零模型前，该网络的层间富人俱乐部系数值：{lm.rcValue(G, file1_m, file2_n)}')
    # print(f'未生成富人俱乐部零模型前，该网络的层间Pearson系数值：{lm.pearson_correlation_coefficient(G, N)}')

    richClubG = lm.isRichClubTwo_0_model(G, N, k1=file1_m, k2=file2_n,  rich=rich, nswap=8, max_tries=40000000)
    print(f'已生成富人俱乐部零模型后，该网络的层间富人俱乐部系数值：{lm.rcValue(richClubG, file1_m, file2_n)}')
    # print(f'未生成富人俱乐部零模型后，该网络的层间Pearson系数值：{lm.pearson_correlation_coefficient(richClubG, N)}')
    df_G_rich = lm.multiGraph_to_csv(richClubG)
    filename = 'two_rich'
    filename += 'up' if rich else 'down'
    filename += '.csv'
    df_G_rich.to_csv(filename, index=False)

# 测试
# inter_richClubTwo_startAndEnd('D:\python\multi_layer_networks\BA.csv', 'D:\python\multi_layer_networks\ws.csv', 7, 6, False)
# inter_richClubTwo_startAndEnd('IEEE.csv', 'Internet.csv', 4, 11, True)
# inter_richClubTwo_startAndEnd('BA_richup--------0.8614.csv', 'ws.csv', 7, 6, False)
# inter_richClubTwo_startAndEnd('BA_pdup.csv', 'ws.csv', 7, 6, False)
# inter_richClubTwo_startAndEnd('BA.csv', 'ws.csv', 7, 6, False)
#     富人
# 0   0.1190
# 5   0.0867
# 8   0.0681
# 10  0.0562
# 15  0.0273
# 20  0.0



# 0 0.0976
# 5 0.25
# 10 0.4516
# 17 0.875


# 对一个网络，如BA或Ws网络，这个单层网络，求其层内的富人俱乐部系数以及生成的新富人俱乐部性质的零模型网络
def intra_richClubSingle_startAndEnd(filename, file_mk, rich=True, nswap=1, max_tries=40000000):
    df = pd.read_csv(filename)

    G = lm.creat_Graph(df)
    start_hubs = [e for e in G.nodes() if G.degree()[e] > file_mk]
    start_hubs_edges = [e for e in G.edges() if G.degree()[e[0]] > file_mk and G.degree()[e[1]] > file_mk]
    start_len_possible_edges = len(start_hubs) * (len(start_hubs) - 1) / 2
    print(f'未生成富人俱乐部零模型前，该网络的层内富人俱乐部系数值：', len(start_hubs_edges) / start_len_possible_edges)

    df_richG = lm.isRichClubSingle_0_model(G, file_mk, rich, nswap, max_tries, connected=1)
    start_hubs1 = [e for e in df_richG.nodes() if df_richG.degree()[e] > file_mk]
    start_hubs_edges1 = [e for e in df_richG.edges() if df_richG.degree()[e[0]] > file_mk and df_richG.degree()[e[1]] > file_mk]
    start_len_possible_edges1 = len(start_hubs1) * (len(start_hubs1) - 1) / 2
    # 找还有多少个节点还有多余的
    for node in start_hubs1:
        neigh_nodes = [e for e in list(df_richG[node]) if df_richG.degree(e) <= file_mk]
        len_neigh_nodes = len(neigh_nodes)
        if len_neigh_nodes != 0:
            print(f'该{node}节点还剩下{len_neigh_nodes}相连节点')
    print(f'已生成富人俱乐部零模型后，该网络的层内富人俱乐部系数值：', len(start_hubs_edges1) / start_len_possible_edges1)
    # print(start_hubs_edges1)
    df_G_rich = lm.graph_to_csv(df_richG)
    filename = 'single_rich'
    filename += 'up' if rich else 'down'
    filename += '.csv'
    df_G_rich.to_csv(filename, index=False)

# intra_richClubSingle_startAndEnd('ws.csv', 6, rich=False, nswap=5000)
# intra_richClubSingle_startAndEnd('Internet.csv', 11, rich=True, nswap=301)

# 对两个网络，如BA和Ws网络，也就是双层网络，先对两个网络进行层内富人俱乐部，然后再进行层间富人。最后求出生成的零模型网络的富人俱乐部系数
def intra_inter_richClubTwo_startAndEnd(filename1, filename2, file1_m, file2_n, richBA=True, richws=True):
    # dfBA = pd.read_csv(filename1)
    # dfws = pd.read_csv(filename2)
    #
    # BA_G = lm.creat_Graph(dfBA)
    # ws_G = lm.creat_Graph(dfws)
    # # 分别对两个网络进行富人俱乐部转换
    # rich_BA_G, coefficient = lm.isRichClubSingle_0_model(BA_G, file1_m, richws, 10000, 400000, connected=1)
    # rich_ws_G, coefficient = lm.isRichClubSingle_0_model(ws_G, file2_n, richws, 10000, 400000, connected=1)
    #
    # # 转换成csv文件
    # filename = 'single_rich'
    # filename += 'up' if richws else 'down'
    # filename += '.csv'
    # df_BAG_rich = lm.graph_to_csv(rich_BA_G)
    # df_BAG_rich.to_csv('1' + filename, index=False)
    # df_wsG_rich = lm.graph_to_csv(rich_ws_G)
    # df_wsG_rich.to_csv('2' + filename, index=False)
    #
    # # 生成两个富人模型
    # inter_richClubTwo_startAndEnd('1' + filename, '2' + filename, file1_m, file2_n, richBA)
    inter_richClubTwo_startAndEnd('BA_richup.csv', 'ws_richdown.csv', file1_m, file2_n, richBA)
    # inter_richClubTwo_startAndEnd('BA_richup.csv', 'ws_richup.csv', file1_m, file2_n, richBA)

# BA_richdown_interdown
# intra_inter_richClubTwo_startAndEnd('BA.csv', 'ws.csv', 7, 6, False, False)
# BA_richdown_interup
# intra_inter_richClubTwo_startAndEnd('BA.csv', 'ws.csv', 7, 6, richBA=True, richws=False)
# BA_richup_interup
# intra_inter_richClubTwo_startAndEnd('BA.csv', 'ws.csv', 7, 6, richBA=True, richws=True)
# BA_richup_interdown
# intra_inter_richClubTwo_startAndEnd('BA.csv', 'ws.csv', 7, 6, richBA=False, richws=True)
# IEEE_richdown_interdown
# intra_inter_richClubTwo_startAndEnd('IEEE.csv', 'Internet.csv', 4, 11, richBA=False, richws=False)
# IEEE_richdown_interup
# intra_inter_richClubTwo_startAndEnd('IEEE.csv', 'Internet.csv', 4, 11, richBA=True, richws=False)
# IEEE_richup_interdown
# intra_inter_richClubTwo_startAndEnd('IEEE.csv', 'Internet.csv', 4, 11, richBA=False, richws=True)
# IEEE_richup_interup
# intra_inter_richClubTwo_startAndEnd('IEEE.csv', 'Internet.csv', 4, 11, richBA=True, richws=True)

'''
正负匹配相关性零模型
'''
def inter_matchEffectTwo_startAndEnd(filename1, filename2, file1_m, file2_n, pattern='positive'):
    dfBA = pd.read_csv(filename1)
    dfws = pd.read_csv(filename2)

    G = lm.create_MultilayerGraphFromPandasEdgeList(dfBA, dfws, source_df1='node1', target_df1='node2',
                                                    source_df2='node1',
                                                    target_df2='node2')
    N = int(len(G.nodes) / 2)
    print(f'未生成相关性零模型前，该网络的层间度Pearson相关系数值：{lm.pearson_correlation_coefficient(G, N)}')
    print(f'未生成相关性零模型前，该网络的层间富人俱乐部系数值：{lm.rcValue(G, file1_m, file2_n)}')
    matchG = lm.matchEffect_0_model(G, N, nswap=60, pattern=pattern)
    print(f'已生成相关性零模型后，该网络的层间度Pearson相关系数值：{lm.pearson_correlation_coefficient(matchG, N)}')
    print(f'已生成相关性零模型后，该网络的层间富人俱乐部系数值：{lm.rcValue(matchG, file1_m, file2_n)}')

    df_G_match = lm.multiGraph_to_csv(matchG)
    filename = 'two_match'
    filename += 'positive' if pattern == 'positive' else 'negative'
    filename += '.csv'
    df_G_match.to_csv(filename, index=False)


# inter_matchEffectTwo_startAndEnd('BA_pd.csv', 'ws.csv', 7, 6, 'negative')
inter_matchEffectTwo_startAndEnd('BA.csv', 'ws.csv', 7, 6, 'negative')
#  度相关
# 0   0.0650
# 10  0.0495
# 20  0.0175
# 30  -0.0264
# 60  -0.0719
# 100 -0.2036
# 200 -0.4281
# 307 -0.6683
def intra_matchEffectSingle_startAndEnd(filename, file_mk, pattern='positive', nswap=1, max_tries=40000000):
    df = pd.read_csv(filename)

    G = lm.creat_Graph(df)
    start_hubs = [e for e in G.nodes() if G.degree()[e] > file_mk]
    start_hubs_edges = [e for e in G.edges() if G.degree()[e[0]] > file_mk and G.degree()[e[1]] > file_mk]
    start_len_possible_edges = len(start_hubs) * (len(start_hubs) - 1) / 2
    print(f'未生成相关性零模型前，该网络的层内度相关系数为: {lm.dac_single(G)}')
    print(f'未生成相关性零模型前，该网络的层内富人俱乐部系数值：', len(start_hubs_edges) / start_len_possible_edges)

    df_richG = lm.isMatchEffectSingle_0_model(G, file_mk, pattern, nswap, max_tries, connected=1)
    start_hubs1 = [e for e in df_richG.nodes() if df_richG.degree()[e] > file_mk]
    start_hubs_edges1 = [e for e in df_richG.edges() if df_richG.degree()[e[0]] > file_mk and df_richG.degree()[e[1]] > file_mk]
    start_len_possible_edges1 = len(start_hubs1) * (len(start_hubs1) - 1) / 2
    for u,v in start_hubs_edges1:
        if u == v:
            print(u,v)
    print(start_hubs_edges1)
    print(f'已生成相关性零模型后，该网络的层内度相关系数为: {lm.dac_single(df_richG)}')
    print(f'已生成相关性零模型后，该网络的层内富人俱乐部系数值：', len(start_hubs_edges1) / start_len_possible_edges1)

    df_G_rich = lm.graph_to_csv(df_richG)
    filename = 'single_rich'
    filename += 'up' if pattern == 'positive' else 'down'
    filename += '.csv'
    df_G_rich.to_csv(filename, index=False)

# intra_matchEffectSingle_startAndEnd('Internet.csv', 11, pattern='positive', nswap=1100)
# intra_matchEffectSingle_startAndEnd('ws.csv', 6, pattern='negetive', nswap=300)

def intra_inter_richClubSingle_startAndEnd(filename1, filename2, file1_m, file2_n, pattern='positive'):

    inter_matchEffectTwo_startAndEnd('BA_pdup.csv', 'ws_pddown.csv', file1_m, file2_n, pattern)

# BA_pdup_interdownSXXXXX
# intra_inter_richClubSingle_startAndEnd('BA.csv', 'ws.csv', 7, 6, pattern='negative')
# BA_pdup_interup
# intra_inter_richClubSingle_startAndEnd('BA.csv', 'ws.csv', 7, 6, pattern='positive')
# BA_richdown_interup
# intra_inter_richClubSingle_startAndEnd('BA.csv', 'ws.csv', 7, 6, pattern='positive')
# BA_richdown_interdown
# intra_inter_richClubSingle_startAndEnd('BA.csv', 'ws.csv', 7, 6, pattern='negative')
# IEEE_richdown_interup
# intra_inter_richClubSingle_startAndEnd('IEEE.csv', 'Internet.csv', 4, 11, pattern='positive')
# IEEE_richdown_interdown
# intra_inter_richClubSingle_startAndEnd('IEEE.csv', 'Internet.csv', 4, 11, pattern='negative')
# IEEE_richup_interup
# intra_inter_richClubSingle_startAndEnd('IEEE.csv', 'Internet.csv', 4, 11, pattern='positive')
# IEEE_richup_interdown
# intra_inter_richClubSingle_startAndEnd('IEEE.csv', 'Internet.csv', 4, 11, pattern='negative')



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
    return nx.degree_assortativity_coefficient(G), nx.degree_pearson_correlation_coefficient(G)

# 层间Person度相关系数
def pearson_degree1(G1,G2):
    list1 = G1.degree()
    list2 = G2.degree()
    list_mid = lm.double_nodes(G1,G2)
    d1 = lm.avg_degree_single(G1)
    d2 = lm.avg_degree_single(G2)
    sum_ab = 0
    sum_a1 = 0
    sum_b1 = 0
    for key in list_mid:
        a = list1[key]-d1
        b = list2[key]-d2
        sum_ab += a*b
        sum_a1 += a**2
        sum_b1 += b**2
    # return sum_ab,math.sqrt(sum_a1),math.sqrt(sum_b1),sum_ab/math.sqrt(sum_a1)/math.sqrt(sum_b1)
    return sum_ab/math.sqrt(sum_a1)/math.sqrt(sum_b1)

def pearson_degree2(Graph, N):
    return lm.pearson_correlation_coefficient(Graph, N)

filenameBA = 'BA_pdup_interup------0.6846----0.9184.csv'
filenameWs = 'ws_pddown.csv'
dfBA = pd.read_csv(filenameBA)
dfWs = pd.read_csv(filenameWs)

df_BAG = lm.creat_Graph(dfBA)
df_WsG = lm.creat_Graph(dfWs)

#
# 生成双层网络
df_BAWsG = lm.create_MultilayerGraphFromPandasEdgeList(dfBA, dfWs, source_df1='node1', target_df1='node2', source_df2='node1',
                                                 target_df2='node2')
N = int(len(df_BAWsG.nodes) / 2)
#
# # 联通性
print(f'BA连通性为: {lm.is_connect_single(df_BAG)}')
print(f'Ws连通性为: {lm.is_connect_single(df_WsG)}')
print(f'BA平均度为: {lm.avg_degree_single(df_BAG)}')
print(f'Ws平均度为: {lm.avg_degree_single(df_WsG)}')
print(f'BA平均最短路径为: {lm.average_shorted_pathLength_single(df_BAG)}')
print(f'Ws平均最短路径为: {lm.average_shorted_pathLength_single(df_WsG)}')

#
# print(f'BA层内度相关系数为: {lm.dac_single(df_BAG)}')
# print(f'Ws层内度相关系数为: {lm.dac_single(df_WsG)}')


# # # #
import time
start_time1 = int(time.time() * 1000)
print(f'层间Person度相关系数1为: {pearson_degree1(df_BAG, df_WsG)}')
end_time1 = int(time.time() * 1000)
start_time2 = int(time.time() * 1000)
print(f'层间Person度相关系数2为: {lm.pearson_correlation_coefficient(df_BAWsG, N)}')
end_time2 = int(time.time() * 1000)
print(end_time1 - start_time1)
print(end_time2 - start_time2)



