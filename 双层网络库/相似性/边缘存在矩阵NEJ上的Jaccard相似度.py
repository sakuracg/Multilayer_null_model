# @Time : Created on 2021/7/30 20:38 
# @File : 边缘存在矩阵NEJ上的Jaccard相似度.py
# @Author : xiaozhi
import ling_model as lm
import numpy as np
import pandas as pd
import networkx as nx

# 首先先把txt文件转换为对应格式的csv文件
# for i in range(37):
#     with open('./data/EU-Air/layer' + str(i+1) + '.txt', 'r') as fr:
#         with open('./data/EU-Air-csv/layer' + str(i+1) + '.csv', 'w') as fw:
#             fw.write('node1,node2\n')
#             for line in fr:
#                 strs = line.strip().split(' ')
#                 fw.write(f'{strs[0]},{strs[1]}\n')

def based_community_layer_similarity_by_edge_jaccard(list1, list2):
    mg1 = nx.Graph(list1)
    mg2 = nx.Graph(list2)

    dict_l1 = lm.property_matrix_edges_existence_by_node_no_aligned(mg1)
    dict_l2 = lm.property_matrix_edges_existence_by_node_no_aligned(mg2)
    # p_l1, p_l2 = list(dict_l1.values()), list(dict_l2.values())

    fun_pl = lm.similarity_binary_property__fun(dict_l1, dict_l2, normalized_fun='jaccard')
    print(fun_pl)
    return fun_pl

res_edge_jaccard = []

# import networkx as nx
#
# with open('./data/EU-Air-csv/layer2.csv') as f:
#     for line in f:
#         str = line.strip().split(',')
#         if str[0] == 'node1':
#             continue
#         res_edge_jaccard.append((int(str[0]), int(str[1])))
# G = nx.Graph(res_edge_jaccard)
# G.has_edge()
# print(G)

def read_file_return_tuple(file):
    tuple_list = []
    with open(file) as f:
        for line in f:
            str = line.strip().split(' ')
            tuple_list.append((int(str[0]), int(str[1])))
    return tuple_list


"""
    测试EU-Air数据
"""
# for i in range(0, 37):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i + 1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j + 1) + '.txt'
#         print(f'layer{i + 1}与{j + 1}层的edge_jaccard相似度为：')
#         list1 = read_file_return_tuple(str_layer1)
#         list2 = read_file_return_tuple(str_layer2)
#         f_measure_val = based_community_layer_similarity_by_edge_jaccard(list1, list2)
#         temp_res.append(format(f_measure_val, '.4f'))
#     res_edge_jaccard.append(temp_res)
# #
# with open('data/EU-Air各种度量结果/res_edge_jaccard.txt', 'a') as f:
#     for i in range(len(res_edge_jaccard)):
#         edge_jaccard_val = res_edge_jaccard[i]
#         for j in range(len(edge_jaccard_val)):
#             k = i + 1 + 0
#             f.write(str(k) + ' ' + edge_jaccard_val[j] + '\n')

"""
    测试Auger数据
"""
for i in range(0, 16):
    temp_res = []
    for j in range(i+1, 16):
        str_layer1 = './data/Auger-txt/layer' + str(i + 1) + '.txt'
        str_layer2 = './data/Auger-txt/layer' + str(j + 1) + '.txt'
        print(f'layer{i + 1}与{j + 1}层的edge_jaccard相似度为：')
        list1 = read_file_return_tuple(str_layer1)
        list2 = read_file_return_tuple(str_layer2)
        f_measure_val = based_community_layer_similarity_by_edge_jaccard(list1, list2)
        temp_res.append(format(f_measure_val, '.4f'))
    res_edge_jaccard.append(temp_res)
#
with open('./data/Auger各种度量结果/res_edge_jaccard.txt', 'a') as f:
    for i in range(len(res_edge_jaccard)):
        edge_jaccard_val = res_edge_jaccard[i]
        for j in range(len(edge_jaccard_val)):
            k = i + 1 + 0
            f.write(str(k) + ' ' + edge_jaccard_val[j] + '\n')
