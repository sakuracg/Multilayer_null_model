# @Time : Created on 2021/7/28 10:34 
# @File : 社区检测通用方法比较层相似性.py
# @Author : xiaozhi
from collections import defaultdict
import igraph as ig

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


def get_Graph_nodes(layer_fileName):
    nodes = []
    with open(layer_fileName, 'r') as f:
        for line in f:
            strs = line.strip().split(' ')
            node1 = int(strs[0])
            node2 = int(strs[1])
            if node1 not in nodes:
                nodes.append(node1)
            if node2 not in nodes:
                nodes.append(node2)
    return nodes


def based_community_layer_similarity_by_purity(layer_edge_1, layer_edge_2):

    G1_nodes = get_Graph_nodes(layer_edge_1)
    G2_nodes = get_Graph_nodes(layer_edge_2)
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

"""
    测试EU-Air数据
"""
# purity_res = []
# for i in range(0, 7):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的purity相似度为：')
#         purity = based_community_layer_similarity_by_purity(str_layer1, str_layer2)
#         temp_res.append(format(purity, '.4f'))
#     purity_res.append(temp_res)




# for i in range(7, 12):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的purity相似度为：')
#         purity = based_community_layer_similarity_by_purity(str_layer1, str_layer2)
#         temp_res.append(format(purity, '.4f'))
#     purity_res.append(temp_res)




# for i in range(12, 19):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的purity相似度为：')
#         purity = based_community_layer_similarity_by_purity(str_layer1, str_layer2)
#         temp_res.append(format(purity, '.4f'))
#     purity_res.append(temp_res)





# for i in range(19, 25):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的purity相似度为：')
#         purity = based_community_layer_similarity_by_purity(str_layer1, str_layer2)
#         temp_res.append(format(purity, '.4f'))
#     purity_res.append(temp_res)






# for i in range(25, 37):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的purity相似度为：')
#         purity = based_community_layer_similarity_by_purity(str_layer1, str_layer2)
#         temp_res.append(format(purity, '.4f'))
#     purity_res.append(temp_res)
# with open('res_purity_new_2.txt', 'a') as f:
#     for i in range(len(purity_res)):
#         puritys = purity_res[i]
#         for j in range(len(puritys)):
#             k = i + 1 + 25
#             f.write(str(k) + ' ' + puritys[j] + '\n')




# with open('res_purity_new.txt', 'a') as f:
#     for i in range(len(purity_res)):
#         puritys = purity_res[i]
#         for j in range(len(puritys)):
#             k = i + 1 + 25
#             f.write(str(k) + ' ' + puritys[j] + '\n')
#





# print(purity_res)

"""
    测试Auger数据
"""
purity_res = []
# for i in range(0, 7):
#     temp_res = []
#     for j in range(i+1, 16):
#         str_layer1 = './data/Auger-txt/layer' + str(i + 1) + '.txt'
#         str_layer2 = './data/Auger-txt/layer' + str(j + 1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的purity相似度为：')
#         purity = based_community_layer_similarity_by_purity(str_layer1, str_layer2)
#         temp_res.append(format(purity, '.4f'))
#     purity_res.append(temp_res)
# for i in range(7, 16):
#     temp_res = []
#     for j in range(i+1, 16):
#         str_layer1 = './data/Auger-txt/layer' + str(i + 1) + '.txt'
#         str_layer2 = './data/Auger-txt/layer' + str(j + 1) + '.txt'
#         print(f'layer{i+1}与{j+1}层的purity相似度为：')
#         purity = based_community_layer_similarity_by_purity(str_layer1, str_layer2)
#         temp_res.append(format(purity, '.4f'))
#     purity_res.append(temp_res)
#
# with open('./data/Auger各种度量结果/res_purity_new.txt', 'a') as f:
#     for i in range(len(purity_res)):
#         puritys = purity_res[i]
#         for j in range(len(puritys)):
#             k = i + 1 + 7
#             f.write(str(k) + ' ' + puritys[j] + '\n')



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

    G1_nodes = get_Graph_nodes(layer_edge_1)
    G2_nodes = get_Graph_nodes(layer_edge_2)
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

f_measure_res = []

"""
    测试EU-Air数据
"""
# f_measure1 = based_community_layer_similarity_by_f_measure('./data/EU-Air/layer' + str(1) + '.txt', './data/EU-Air/layer' + str(3) + '.txt')
# f_measure1= based_community_layer_similarity_by_f_measure('./data/EU-Air/layer' + str(1) + '.txt', './data/EU-Air/layer' + str(2) + '.txt')
# # print(f_measure1)
# for i in range(0, 7):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i + 1}与{j + 1}层的f_measure相似度为：')
#         f_measure_val = based_community_layer_similarity_by_f_measure(str_layer1, str_layer2)
#         temp_res.append(format(f_measure_val, '.4f'))
#     f_measure_res.append(temp_res)
#
#
# for i in range(7, 13):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i + 1}与{j + 1}层的f_measure相似度为：')
#         f_measure_val = based_community_layer_similarity_by_f_measure(str_layer1, str_layer2)
#         temp_res.append(format(f_measure_val, '.4f'))
#     f_measure_res.append(temp_res)

#
#
#
#
for i in range(13, 20):
    temp_res = []
    for j in range(i+1, 37):
        str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
        str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
        print(f'layer{i + 1}与{j + 1}层的f_measure相似度为：')
        f_measure_val = based_community_layer_similarity_by_f_measure(str_layer1, str_layer2)
        temp_res.append(format(f_measure_val, '.4f'))
    f_measure_res.append(temp_res)
#
#
#
# #
# for i in range(20, 25):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i + 1}与{j + 1}层的f_measure相似度为：')
#         f_measure_val = based_community_layer_similarity_by_f_measure(str_layer1, str_layer2)
#         temp_res.append(format(f_measure_val, '.4f'))
#     f_measure_res.append(temp_res)
#
#
# for i in range(25, 37):
#     temp_res = []
#     for j in range(i+1, 37):
#         str_layer1 = './data/EU-Air/layer' + str(i+1) + '.txt'
#         str_layer2 = './data/EU-Air/layer' + str(j+1) + '.txt'
#         print(f'layer{i + 1}与{j + 1}层的f_measure相似度为：')
#         f_measure_val = based_community_layer_similarity_by_f_measure(str_layer1, str_layer2)
#         temp_res.append(format(f_measure_val, '.4f'))
#     f_measure_res.append(temp_res)
#
# with open('data/EU-Air各种度量结果/res_f_measure_new.txt', 'a') as f:
#     for i in range(len(f_measure_res)):
#         f_measures = f_measure_res[i]
#         for j in range(len(f_measures)):
#             k = i + 1 + 25
#             f.write(str(k) + ' ' + f_measures[j] + '\n')

"""
    测试Auger数据
"""


# for i in range(0, 16):
#     temp_res = []
#     for j in range(i+1, 16):
#         str_layer1 = './data/Auger-txt/layer' + str(i + 1) + '.txt'
#         str_layer2 = './data/Auger-txt/layer' + str(j + 1) + '.txt'
#         print(f'layer{i + 1}与{j + 1}层的f_measure相似度为：')
#         f_measure_val = based_community_layer_similarity_by_f_measure(str_layer1, str_layer2)
#         temp_res.append(format(f_measure_val, '.4f'))
#     f_measure_res.append(temp_res)
#
# with open('data/Auger各种度量结果/res_f_measure_new.txt', 'a') as f:
#     for i in range(len(f_measure_res)):
#         f_measures = f_measure_res[i]
#         for j in range(len(f_measures)):
#             k = i + 1 + 0
#             f.write(str(k) + ' ' + f_measures[j] + '\n')




