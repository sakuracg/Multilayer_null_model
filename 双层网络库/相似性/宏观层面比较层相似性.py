# @Time : Created on 2021/8/2 10:52 
# @File : 宏观层面比较层相似性.py 
# @Author : xiaozhi

import numpy as np
import networkx as nx
import ling_model as lm

# 平均节点度
# mg1 = nx.read_edgelist('./data/EU-Air/layer1.txt')
# mg2 = nx.read_edgelist('./data/EU-Air/layer2.txt')

def based_community_layer_similarity_by_nodeDegree_Avg(str_layer1, str_layer2):
    mg1 = nx.read_edgelist(str_layer1)
    mg2 = nx.read_edgelist(str_layer2)

    pl1 = list(dict(mg1.degree()).values())
    pl2 = list(dict(mg2.degree()).values())

    mean_p1 = lm.property_matrix_get_avg_val_fun(P_L=pl1)
    mean_p2 = lm.property_matrix_get_avg_val_fun(P_L=pl2)

    node_degree_avg = lm.property_matrix_relative_difference_fun(mean_p1, mean_p2)
    return node_degree_avg

# va = based_community_layer_similarity_by_nodeDegree_Avg('./data/EU-Air/layer1.txt', './data/EU-Air/layer2.txt')
# print(va)
"""
    测试EU-Air数据
"""
# mean_res = []
# for i in range(0, 37):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的mean平均值的相似度为：')
#         cos_val = based_community_layer_similarity_by_nodeDegree_Avg(str_layer1, str_layer2)
#         temp_res.append(format(cos_val, '.4f'))
#     mean_res.append(temp_res)
# with open('res_degree_mean_val.txt', 'a') as f:
#     for i in range(len(mean_res)):
#         mean_vals = mean_res[i]
#         for j in range(len(mean_vals)):
#             k = i + 1 + 0
#             f.write(str(k) + ' ' + mean_vals[j] + '\n')

"""
    测试Auger数据
"""
# mean_res = []
# for i in range(0, 16):
#     temp_res = []
#     for j in range(i+1, 16):
#         str_layer1 = './data/Auger-txt/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/Auger-txt/layer' + str(j+1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的mean平均值的相似度为：')
#         cos_val = based_community_layer_similarity_by_nodeDegree_Avg(str_layer1, str_layer2)
#         temp_res.append(format(cos_val, '.4f'))
#     mean_res.append(temp_res)
# with open('./data/Auger各种度量结果/res_degree_mean_val.txt', 'a') as f:
#     for i in range(len(mean_res)):
#         mean_vals = mean_res[i]
#         for j in range(len(mean_vals)):
#             k = i + 1 + 0
#             f.write(str(k) + ' ' + mean_vals[j] + '\n')


# 平均聚类系数
def based_community_layer_similarity_by_nodeCC_Avg(str_layer1, str_layer2):
    mg1 = nx.read_edgelist(str_layer1)
    mg2 = nx.read_edgelist(str_layer2)

    pl1 = lm.property_matrix_nodes_CC(mg1)
    pl2 = lm.property_matrix_nodes_CC(mg2)

    mean_p1 = lm.property_matrix_get_avg_val_fun(P_L=pl1)
    mean_p2 = lm.property_matrix_get_avg_val_fun(P_L=pl2)

    node_CC_avg = lm.property_matrix_relative_difference_fun(mean_p1, mean_p2)
    return node_CC_avg

# print(based_community_layer_similarity_by_nodeCC_Avg('./data/EU-Air/layer4.txt', './data/EU-Air/layer9.txt'))
"""
    测试EU-Air数据
"""
# mean_res = []
# for i in range(0, 37):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的meanCC平均值的相似度为：')
#         cos_val = based_community_layer_similarity_by_nodeCC_Avg(str_layer1, str_layer2)
#         temp_res.append(format(cos_val, '.4f'))
#     mean_res.append(temp_res)
# with open('res_cc_mean_val.txt', 'a') as f:
#     for i in range(len(mean_res)):
#         mean_vals = mean_res[i]
#         for j in range(len(mean_vals)):
#             k = i + 1 + 0
#             f.write(str(k) + ' ' + mean_vals[j] + '\n')


"""
    测试Auger数据
"""

mean_res = []
for i in range(0, 16):
    temp_res = []
    for j in range(i+1, 16):
        str_layer1 = './data/Auger-txt/layer' + str(i + 1) + '.txt'
        str_layer2 = './data/Auger-txt/layer' + str(j+1) + '.txt'
        print(f'layer{i+1}与{j+1}层的meanCC平均值的相似度为：')
        cos_val = based_community_layer_similarity_by_nodeCC_Avg(str_layer1, str_layer2)
        temp_res.append(format(cos_val, '.4f'))
    mean_res.append(temp_res)
with open('./data/Auger各种度量结果/res_cc_mean_val.txt', 'a') as f:
    for i in range(len(mean_res)):
        mean_vals = mean_res[i]
        for j in range(len(mean_vals)):
            k = i + 1 + 0
            f.write(str(k) + ' ' + mean_vals[j] + '\n')
