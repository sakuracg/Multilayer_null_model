# @Time : Created on 2021/8/2 16:07 
# @File : 微观层面比较层相似性.py 
# @Author : xiaozhi

import numpy as np
import networkx as nx
import ling_model as lm


# 节点度矩阵上的余弦相似度

def based_community_layer_similarity_change_vector_differToSame(mg1, mg2):
    """
    由于dict中的长度是不相同的，在进行向量计算时把对应缺少的位置补0
    然后按照dict_l1，dict_l2的键，取出对应节点的度值并放进两个列表中
    :param mg1:
    :param mg2:
    :return:
    """
    dict_l1 = {k: v for k, v in nx.degree(mg1)}
    dict_l2 = {k: v for k, v in nx.degree(mg2)}
    for key in dict_l1:
        if dict_l2.get(key) == None:
            dict_l2[key] = 0

    for key in dict_l2:
        if dict_l1.get(key) == None:
            dict_l1[key] = 0

    # 然后按照dict_l1，dict_l2的键，取出对应节点的度值并放进两个列表中
    P_list_l1 = []
    P_list_l2 = []
    for key in sorted(dict_l1):
        P_list_l1.append(dict_l1[key])
        P_list_l2.append(dict_l2[key])

    return P_list_l1, P_list_l2


# 传入两层的节点度
def based_community_layer_similarity_by_nodeDegree_cos(str_layer1, str_layer2):
    """
    节点度矩阵 (NDC) 上的余弦相似度
    :param str_layer1:
    :param str_layer2:
    :return:
    """
    # 按照文本文件的固定格式读取网络
    mg1 = nx.read_edgelist(str_layer1)
    mg2 = nx.read_edgelist(str_layer2)
    # 由于dict中的长度是不相同的，在进行向量计算时把对应缺少的位置补0
    # 补全向量
    P_list_l1, P_list_l2 = based_community_layer_similarity_change_vector_differToSame(mg1, mg2)
    np_p1 = np.array(P_list_l1)
    np_p2 = np.array(P_list_l2)

    molecule = np.sum(np_p1 * np_p2)
    sum_p1 = np.sqrt(np.sum([p * p for p in P_list_l1]))
    sum_p2 = np.sqrt(np.sum([p * p for p in P_list_l2]))
    denominator = sum_p1 * sum_p2
    return molecule / denominator


# cos_val = based_community_layer_similarity_by_nodeDegree_cos(mg1,mg2)
"""
    测试EU-Air数据
"""
# cos_res = []
# for i in range(0, 37):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的purity相似度为：')
#         cos_val = based_community_layer_similarity_by_nodeDegree_cos(str_layer1, str_layer2)
#         temp_res.append(format(cos_val, '.4f'))
#     cos_res.append(temp_res)
# with open('res_cos_val.txt', 'a') as f:
#     for i in range(len(cos_res)):
#         cos_vals = cos_res[i]
#         for j in range(len(cos_vals)):
#             k = i + 1 + 0
#             f.write(str(k) + ' ' + cos_vals[j] + '\n')

"""
    测试Auger数据
"""
# cos_res = []
# for i in range(0, 16):
#     temp_res = []
#     for j in range(i+1, 16):
#         str_layer1 = './data/Auger-txt/layer' + str(i + 1) + '.txt'
#         str_layer2 = './data/Auger-txt/layer' + str(j + 1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的purity相似度为：')
#         cos_val = based_community_layer_similarity_by_nodeDegree_cos(str_layer1, str_layer2)
#         temp_res.append(format(cos_val, '.4f'))
#     cos_res.append(temp_res)
# with open('./data/Auger各种度量结果/res_cos_val.txt', 'a') as f:
#     for i in range(len(cos_res)):
#         cos_vals = cos_res[i]
#         for j in range(len(cos_vals)):
#             k = i + 1 + 0
#             f.write(str(k) + ' ' + cos_vals[j] + '\n')

# 节点度矩阵上的Pearson相关系数
def based_community_layer_similarity_by_nodeDegree_Pearson(str_layer1, str_layer2):
    """
    TODO
    :param str_layer1:
    :param str_layer2:
    :return:
    """
    mg1 = nx.read_edgelist(str_layer1)
    mg2 = nx.read_edgelist(str_layer2)
    # 由于dict中的长度是不相同的，在进行向量计算时把对应缺少的位置补0
    P_list_l1, P_list_l2 = based_community_layer_similarity_change_vector_differToSame(mg1, mg2)
    np_p1 = np.array(P_list_l1)
    np_p2 = np.array(P_list_l2)

    mean_val1 = lm.property_matrix_get_avg_val_fun(np_p1)
    mean_val2 = lm.property_matrix_get_avg_val_fun(np_p2)

    new_np_p1 = np_p1 - mean_val1
    new_np_p2 = np_p2 - mean_val2

    molecule = np.sum(new_np_p1 * new_np_p2)
    sum_p1 = np.sqrt(np.sum([p * p for p in new_np_p1]))
    sum_p2 = np.sqrt(np.sum([p * p for p in new_np_p2]))
    denominator = sum_p1 * sum_p2
    return molecule / denominator

# cos_val = based_community_layer_similarity_by_nodeDegree_Pearson('./data/EU-Air/layer3.txt','./data/EU-Air/layer2.txt')
# print(cos_val)
"""
    测试EU-Air数据
"""
# cos_res = []
# for i in range(0, 37):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的purity相似度为：')
#         cos_val = based_community_layer_similarity_by_nodeDegree_Pearson(str_layer1, str_layer2)
#         temp_res.append(format(cos_val, '.4f'))
#     cos_res.append(temp_res)
# with open('data/EU-Air各种度量结果/res_pearson_val.txt', 'a') as f:
#     for i in range(len(cos_res)):
#         cos_vals = cos_res[i]
#         for j in range(len(cos_vals)):
#             k = i + 1 + 0
#             f.write(str(k) + ' ' + cos_vals[j] + '\n')
"""
    测试Auger数据
"""
cos_res = []
for i in range(0, 16):
    temp_res = []
    for j in range(i+1, 16):
        str_layer1 = './data/Auger-txt/layer' + str(i + 1) + '.txt'
        str_layer2 = './data/Auger-txt/layer' + str(j + 1) + '.txt'
        print(f'layer{i+1}与{j+1}层的purity相似度为：')
        cos_val = based_community_layer_similarity_by_nodeDegree_Pearson(str_layer1, str_layer2)
        temp_res.append(format(cos_val, '.4f'))
    cos_res.append(temp_res)
with open('./data/Auger各种度量结果/res_pearson_val.txt', 'a') as f:
    for i in range(len(cos_res)):
        cos_vals = cos_res[i]
        for j in range(len(cos_vals)):
            k = i + 1 + 0
            f.write(str(k) + ' ' + cos_vals[j] + '\n')


# 边缘存在矩阵的Jaccard相似度
# 已经计算
