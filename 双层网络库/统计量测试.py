# @Time : Created on 2021/5/10 16:48 
# @File : 统计量测试.py
# @Author : xiaozhi

import ling_model as lm
import numpy as np
import pandas as pd
import math
import networkx as nx
import multinetx as mx
# N = 8
# mg = lm.create_MultilayerGraph(N, p_g1=0.3, p_g2=0.4, seed_g1=201, seed_g2=660, intra_layer_edges_weight=2, inter_layer_edges_weight=3)
# N = int(len(mg.nodes) / 2)
# dfBA = pd.read_csv('./data/EU-Air/layer1.txt')
# dfws = pd.read_csv('./data/EU-Air/layer2.txt')
# dfBA = pd.read_csv('BA.csv')
# dfws = pd.read_csv('ws.csv')

mg1 = nx.read_edgelist('./data/EU-Air/layer1.txt')
mg2 = nx.read_edgelist('./data/EU-Air/layer2.txt')
# mg = lm.create_MultilayerGraphFromPandasEdgeList(dfBA, dfws, source_df1='node1', target_df1='node2', source_df2='node1',
#                                                  target_df2='node2')

# N1 = int(len(mg.nodes) / 2)
# N2 = int(len(mg.nodes) / 2)
# print(N)
# print(mg.edges)
# # a.values()
# # print(lm.property_matrix_nodes_degree(mg, N))
# # print(lm.property_matrix_nodes_CC(mg, N))
# dict_l1, dict_l2 = lm.property_matrix_edges_existence(mg, N)
# dict_l1, dict_l2 = lm.property_matrix_triangles_existence(mg, N)
# p_l1, p_l2 = list(dict_l1.values()), list(dict_l2.values())
# p_l1, p_l2 = lm.property_matrix_nodes_CC(mg, N)
p_l1 = lm.property_matrix_nodes_degree(mg1)
p_l2 = lm.property_matrix_nodes_degree(mg2)
print(p_l1)
print(p_l2)
fun_pl1 = lm.property_matrix_get_avg_val_fun(p_l1)
fun_pl2 = lm.property_matrix_get_avg_val_fun(p_l2)
print(fun_pl1)
print(fun_pl2)
print(lm.property_matrix_relative_difference_fun(fun_pl1, fun_pl2))
# print(p_l1)
# print('\n\n')
# print(p_l2)
# print(lm.similarity_binary_property__fun(p_l1, p_l2, normalized_fun='jaccard'))
# print(lm.similarity_binary_property__fun(p_l1, p_l2, normalized_fun='smc'))
# print(lm.similarity_binary_property__fun(p_l1, p_l2, normalized_fun='hamann'))
# print(lm.similarity_binary_property__fun(p_l1, p_l2, normalized_fun='kulczynski'))
# print(lm.property_matrix_triangles_existence(mg, N))
# print(lm.similarity_binary_property__fun(p_l1, p_l2))

# a = np.array([1,2,3])
# print(np.power(a, 3))

# dfBA = pd.read_csv('BA.csv')
# dfws = pd.read_csv('ws.csv')
#
# mg = lm.create_MultilayerGraphFromPandasEdgeList(dfBA, dfws, source_df1='node1', target_df1='node2', source_df2='node1',
#                                                  target_df2='node2')
# N = int(len(mg.nodes) / 2)
# dict_l1, dict_l2 = lm.property_matrix_edges_existence(mg, N)
# p_l1, p_l2 = list(dict_l1.values()), list(dict_l2.values())
# print(lm.similarity_binary_property__fun(p_l1, p_l2, normalized_fun='jaccard'))
# print(lm.similarity_binary_property__fun(p_l1, p_l2, normalized_fun='smc'))
# print(lm.similarity_binary_property__fun(p_l1, p_l2, normalized_fun='hamann'))
# print(lm.similarity_binary_property__fun(p_l1, p_l2, normalized_fun='russelrao'))

# c = lm.overall_of_overlap_network(mg, N)
#
# newGraph = lm.matchEffect_0_model(mg, N, nswap=2000, pattern='positive', max_tries= 2000)
# # richGraph = lm.isRichClubTwo_0_model(mg, N, k1=4, k2=3, rich=True, nswap=70, max_tries=70)
# dict_l1, dict_l2 = lm.property_matrix_edges_existence(newGraph, N)
# p_l1, p_l2 = list(dict_l1.values()), list(dict_l2.values())
# print(lm.similarity_binary_property__fun(p_l1, p_l2))
#
# c = lm.pearson_correlation_coefficient_byLongVector(mg)
# c = lm.mutual_information_byLongVector(mg)
# print(c)
# newGraph = lm.matchEffect_0_model(mg, N, nswap=2000, pattern='negative')
#
# rc = lm.overall_of_overlap_network(newGraph, N)

# c1 = lm.pearson_correlation_coefficient_byLongVector(newGraph)
# c1 = lm.mutual_information_byLongVector(newGraph)
# print(c)
# print(c1)