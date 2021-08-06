# @Time : Created on 2021/7/25 16:20 
# @File : 社区检测比较层相似性.py
# @Author : xiaozhi
from collections import defaultdict

import networkx as nx
import networkx.algorithms as nal
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt

# ./data/EU-air/layer1
# df = pd.read_csv('IEEE-----------0.0809.csv')
#
# G = nx.from_pandas_edgelist(df,'node1','node2')
# nx.draw(G,with_labels=True)
# plt.show()
#
# c = list(nal.community.greedy_modularity_communities(G))
# G = nx.read_edgelist('test1.txt', nodetype=int, encoding='utf-8')
# Gi_1 = ig.Graph.Read_Edgelist('test1.txt')
# Gi_1 = Gi_1.subgraph(map(int, G.nodes()))  # G.nodes()获取一个Graph对象中的node数组,
# G1 = Gi_1.as_undirected()
#
# G = nx.read_edgelist('test2.txt', nodetype=int, encoding='utf-8')
# Gi_2 = ig.Graph.Read_Edgelist('test2.txt')
# Gi_2 = Gi_2.subgraph(map(int, G.nodes()))  # G.nodes()获取一个Graph对象中的node数组,
# G2 = Gi_2.as_undirected()
# 处理方法是读取文件并把单个节点的社区对比文件中的，然后看看在不在列表中，如果不在则continue
# G1_nodes = []
# with open('test1.txt', 'r') as f:
#     for line in f:
#         strs = line.strip().split(' ')
#         node1 = int(strs[0])
#         node2 = int(strs[1])
#         if node1 not in G1_nodes:
#             G1_nodes.append(node1)
#         if node2 not in G1_nodes:
#             G1_nodes.append(node2)
# G2_nodes = []
# with open('test2.txt', 'r') as f:
#     for line in f:
#         strs = line.strip().split(' ')
#         node1 = int(strs[0])
#         node2 = int(strs[1])
#         if node1 not in G2_nodes:
#             G2_nodes.append(node1)
#         if node2 not in G2_nodes:
#             G2_nodes.append(node2)
#
# Gi_1 = ig.Graph.Read_Edgelist('test1.txt')
# Gi_2 = ig.Graph.Read_Edgelist('test2.txt')  #基于这些连边使用igraph创建一个新网络
# G1 = Gi_1.as_undirected()
# G2 = Gi_2.as_undirected()
# # 检测到的社区
# community_list1 = G1.community_multilevel(weights=None, return_levels=False)
# community_list2 = G2.community_multilevel(weights=None, return_levels=False)
# print(community_list1[0][0] in [0, 1, 2, 3])
# Q=community_list1.modularity 计算模块度
# print(community_list1)
# print(community_list2)
# for i in range(len(community_list1)):
#     print(community_list1[i])
# print(len(community_list1._membership))
# 计算purity纯度

# 首先得到每一层中的社区集合以及集合中节点的数量
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
    # print(sum_k)
    return sum_k / len(G1_nodes)

def get_list_intersection_length(list1, list2):
    return len(set(list1).intersection(set(list2)))

# community_list1 = [[1,2,3,4],[5,6,7,8,9],[10,11,12,13,14,15],[16,17,18]]
# community_list2 = [[1, 2, 3, 19, 20], [4, 5, 7, 11],[6,8,9],[10,12,13,14,15]]

# purity1 = purity_single(community_list1, community_list2, G1_nodes, G2_nodes)
# purity2 = purity_single(community_list2, community_list1, G2_nodes, G1_nodes)
# purity = 2 * purity1 * purity2 / (purity1 + purity2)
# print(f'计算purity的纯度{purity}')



# ·································································

# 计算F—Measure

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
        print(f'prec{max_cun}/{len(temp_list)}recall{max_cun}/{m_ji}')
        sum_f += f_i
    # print(len_community_real)
    return sum_f / len_community_real

# f_measure_1 = f_measure(community_list1, community_list2, G1_nodes, G2_nodes)
# f_measure_2 = f_measure(community_list2, community_list1, G2_nodes, G1_nodes)
f_measure_1 = f_measure( [[1,2,3,4],[5,6,7,8,9],[10,11,12,13,14,15],[16,17,18]],
                         [[1, 2, 3, 19, 20], [4, 5, 7, 11],[6,8,9],[10,12,13,14,15]],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,19,20])
f_measure_2 = f_measure([[1, 2, 3, 19, 20], [4, 5, 7, 11],[6,8,9],[10,12,13,14,15]],
                        [[1,2,3,4],[5,6,7,8,9],[10,11,12,13,14,15],[16,17,18]],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,19,20],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])


f_val = 2 * f_measure_1 * f_measure_2 / (f_measure_1 + f_measure_2)
# print(f_measure_1)
# print(f_measure_2)
# print(f_val)
# print(f_measure(community_list1, community_list2))
# print(f_measure([[8, 14, 15, 18, 20, 22, 23, 26, 27, 29, 30, 32, 33], [0, 1, 2, 3, 7, 9, 11, 12, 13, 17, 19, 21], [4, 5, 6, 10, 16], [24, 25, 28, 31]],
#              [[8, 9, 14, 15, 18, 20, 22, 26, 29, 30, 32, 33], [0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21], [23, 24, 25, 27, 28, 31], [4, 5, 6, 10, 16]]))
# print(f_measure())