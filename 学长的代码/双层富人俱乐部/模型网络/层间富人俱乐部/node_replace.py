# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:43:32 2019

@author: PCC
"""

import pandas as pd
import networkx as nx
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import copy
from collections import Counter











def get_keys(d, value):
    return [k for k,v in d.items() if v == value]


'''
双通节点
'''
def double_nodes(G_call,G_sms):
    return list(set(G_call.nodes()).intersection(set(G_sms.nodes()))) # 交集

'''
删除字典中value值小于等于1 的key—value
'''
def delete_kv(c):
    for k in list(c.keys()):
        if c[k] <=1:
            del c[k]

'''
将字典中key值为a和b互换位置（value值位置不变）
'''            
def change(mydict,a,b):
    list1 = list(mydict.keys())
    list2 = list(mydict.values())
    ia = list1.index(a)
    ib = list1.index(b)
    list1[ia], list1[ib] = list1[ib], list1[ia]
    mydict_new = dict(zip(list1, list2))
    return mydict_new
      


                  
            
'''
层间度相关系数正向（保证层间活跃度相关系数不变）

创建key值为节点，value值为强度的字典
将强度相同的所有节点分别放入对应的list
随机取出某个强度list中的两个节点，比较公式结果值
若增加则将原始度分布的两个节点的度值互换（之后都在互换过的度分布字典中进行互换）
且将两个节点的标号互换
'''

def replace_degree_in0(G0,G_sms, nswap=1, max_tries=100):
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")  
    n = 0
    swapcount = 0
    G = copy.deepcopy(G0)   
    node_list=list(G0.nodes()) 
    node_list_shuffling=node_list[:]   
    list_s = [G0.degree(i,weight = 'weight') for i in double_nodes(G0,G_sms)]
    dict_s = dict(zip(double_nodes(G0,G_sms),list_s))
    d1 = Counter(list_s)
    delete_kv(d1)
    dict_degree = dict(G.degree())                     
    while swapcount < nswap:
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        list1 = get_keys(dict_s,random.sample(list(d1.keys()),1)[0])   
        node1,node2 = random.sample(list1,2)
        degree_G1n1, degree_G1n2 = dict_degree[node1],dict_degree[node2]
        degree_G2n1, degree_G2n2 = dict(G_sms.degree([node1,node2])).values()
        n += 1

        if (degree_G1n1-degree_G1n2)*(degree_G2n1-degree_G2n2) >= 0:
            continue
        else:
            dict_degree = change(dict_degree,node1,node2)
            ia  =node_list_shuffling.index(node1)
            ib  =node_list_shuffling.index(node2)
            node_list_shuffling[ia], node_list_shuffling[ib] = node_list_shuffling[ib], node_list_shuffling[ia]
            swapcount += 1
            print('Valued numbers',swapcount)
    node_dict = dict(zip(node_list,node_list_shuffling))
    G = nx.relabel_nodes(G, node_dict)
    return G



'''
层间度相关系数负向（保证层间活跃度相关系数不变）
'''


def replace_degree_de0(G0,G_sms, nswap=1, max_tries=100):
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")  
    n = 0
    swapcount = 0
    G = copy.deepcopy(G0)   
    node_list=list(G0.nodes()) 
    node_list_shuffling=node_list[:]   
    list_s = [G0.degree(i,weight = 'weight') for i in double_nodes(G0,G_sms)]
    dict_s = dict(zip(double_nodes(G0,G_sms),list_s))
    d1 = Counter(list_s)
    delete_kv(d1)
    dict_degree = dict(G.degree())                     
    while swapcount < nswap:
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        list1 = get_keys(dict_s,random.sample(list(d1.keys()),1)[0])   
        node1,node2 = random.sample(list1,2)
        degree_G1n1, degree_G1n2 = dict_degree[node1],dict_degree[node2]
        degree_G2n1, degree_G2n2 = dict(G_sms.degree([node1,node2])).values()
        n += 1
        if (degree_G1n1-degree_G1n2)*(degree_G2n1-degree_G2n2) <= 0:
            continue
        else:
            dict_degree = change(dict_degree,node1,node2)
            ia  =node_list_shuffling.index(node1)
            ib  =node_list_shuffling.index(node2)
            node_list_shuffling[ia], node_list_shuffling[ib] = node_list_shuffling[ib], node_list_shuffling[ia]
            swapcount += 1
            print('Valued numbers',swapcount)  
    node_dict = dict(zip(node_list,node_list_shuffling))
    G = nx.relabel_nodes(G, node_dict)
    return G





##################################################################################################################################################           
            
'''
层间活跃度度相关系数正向（保证层间度相关系数不变）
'''           
def replace_degree_in1(G0,G_sms, nswap=1, max_tries=100):
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")  
    n = 0
    swapcount = 0
    G = copy.deepcopy(G0)   
    node_list=list(G0.nodes()) 
    node_list_shuffling=node_list[:]   
    list_s = [G0.degree(i) for i in double_nodes(G0,G_sms)]
    dict_s = dict(zip(double_nodes(G0,G_sms),list_s))
    d1 = Counter(list_s)
    delete_kv(d1)
    dict_degree = dict(zip(G0.nodes(),[G0.degree(i,weight = 'weight') for i in G0.nodes()]))       
    dict_degree_sms = dict(zip(G_sms.nodes(),[G_sms.degree(i,weight = 'weight') for i in G_sms.nodes()]))                   
    while swapcount < nswap:
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        list1 = get_keys(dict_s,random.sample(list(d1.keys()),1)[0])   
        node1,node2 = random.sample(list1,2)
        degree_G1n1, degree_G1n2 = dict_degree[node1],dict_degree[node2]
        degree_G2n1, degree_G2n2 = dict_degree_sms[node1],dict_degree_sms[node2]
        n += 1
        if (degree_G1n1-degree_G1n2)*(degree_G2n1-degree_G2n2) >= 0:
            continue
        else:
            dict_degree = change(dict_degree,node1,node2)
            ia  =node_list_shuffling.index(node1)
            ib  =node_list_shuffling.index(node2)
            node_list_shuffling[ia], node_list_shuffling[ib] = node_list_shuffling[ib], node_list_shuffling[ia]
            swapcount += 1
            print('Valued numbers',swapcount)    
    node_dict = dict(zip(node_list,node_list_shuffling))
    G = nx.relabel_nodes(G, node_dict)
    return G          
            
'''
层间活跃度度相关系数负向（保证层间度相关系数不变）
''' 
def replace_degree_de1(G0,G_sms, nswap=1, max_tries=100):
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")  
    n = 0
    swapcount = 0
    cnt = 0
    G = copy.deepcopy(G0)   
    node_list=list(G0.nodes()) 
    node_list_shuffling=node_list[:]   
    list_s = [G0.degree(i) for i in double_nodes(G0,G_sms)]
    dict_s = dict(zip(double_nodes(G0,G_sms),list_s))
    d1 = Counter(list_s)
    delete_kv(d1)
    dict_degree = dict(zip(G0.nodes(),[G0.degree(i,weight = 'weight') for i in G0.nodes()]))      
    dict_degree_sms = dict(zip(G_sms.nodes(),[G_sms.degree(i,weight = 'weight') for i in G_sms.nodes()]))                                      
    while swapcount < nswap:
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        cnt+=1
        print(cnt)
        list1 = get_keys(dict_s,random.sample(list(d1.keys()),1)[0])   
        node1,node2 = random.sample(list1,2)
        degree_G1n1, degree_G1n2 = dict_degree[node1],dict_degree[node2]
        degree_G2n1, degree_G2n2 = dict_degree_sms[node1],dict_degree_sms[node2]
        n += 1
        if (degree_G1n1-degree_G1n2)*(degree_G2n1-degree_G2n2) <= 0:
            continue
        else:
            dict_degree = change(dict_degree,node1,node2)
            ia  =node_list_shuffling.index(node1)
            ib  =node_list_shuffling.index(node2)
            node_list_shuffling[ia], node_list_shuffling[ib] = node_list_shuffling[ib], node_list_shuffling[ia]
            swapcount += 1
    print('Valued numbers',swapcount)    
    node_dict = dict(zip(node_list,node_list_shuffling))
    G = nx.relabel_nodes(G, node_dict)
    return G                      
##################################################################################################################################################           
            
            
'''
度相关系数正向（不保证层间度相关系数不变）

'''

def replace_degree_in2(G0,G_sms, nswap=1, max_tries=100):
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")  
    n = 0
    swapcount = 0
    G = copy.deepcopy(G0)   
    node_list=list(G0.nodes()) 
    node_list_shuffling=node_list[:]   
    list_mid = double_nodes(G0,G_sms)

    dict_degree = dict(G.degree())                     
    while swapcount < nswap:
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        node1,node2 = random.sample(list_mid,2)
        degree_G1n1, degree_G1n2 = dict_degree[node1],dict_degree[node2]
        degree_G2n1, degree_G2n2 = dict(G_sms.degree([node1,node2])).values()
        n += 1
        if (degree_G1n1-degree_G1n2)*(degree_G2n1-degree_G2n2) >= 0:
            continue
        else:
            dict_degree = change(dict_degree,node1,node2)
            ia  =node_list_shuffling.index(node1)
            ib  =node_list_shuffling.index(node2)
            node_list_shuffling[ia], node_list_shuffling[ib] = node_list_shuffling[ib], node_list_shuffling[ia]
            swapcount += 1
            print('Valued numbers',swapcount)  
    node_dict = dict(zip(node_list,node_list_shuffling))
    G = nx.relabel_nodes(G, node_dict)
    return G



'''
度相关系数负向（不保证层间度相关系数不变）
'''

def replace_degree_de2(G0,G_sms, nswap=1, max_tries=100):
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")  
    n = 0
    swapcount = 0
    G = copy.deepcopy(G0)   
    node_list=list(G0.nodes()) 
    node_list_shuffling=node_list[:]   
    list_mid = list_mid = double_nodes(G0,G_sms)

    dict_degree = dict(G.degree())                     
    while swapcount < nswap:
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        node1,node2 = random.sample(list_mid,2)
        degree_G1n1, degree_G1n2 = dict_degree[node1],dict_degree[node2]
        degree_G2n1, degree_G2n2 = dict(G_sms.degree([node1,node2])).values()
        n += 1
        if (degree_G1n1-degree_G1n2)*(degree_G2n1-degree_G2n2) <= 0:
            continue
        else:
            dict_degree = change(dict_degree,node1,node2)
            ia  =node_list_shuffling.index(node1)
            ib  =node_list_shuffling.index(node2)
            node_list_shuffling[ia], node_list_shuffling[ib] = node_list_shuffling[ib], node_list_shuffling[ia]
            swapcount += 1
            print('Valued numbers',swapcount)  
    node_dict = dict(zip(node_list,node_list_shuffling))
    G = nx.relabel_nodes(G, node_dict)
    return G


'''
活跃度相关系数正向（不保证层间度相关系数不变）
'''
def replace_degree_in3(G0,G_sms, nswap=1, max_tries=100):
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")  
    n = 0
    swapcount = 0
    G = copy.deepcopy(G0)   
    node_list=list(G0.nodes()) 
    node_list_shuffling=node_list[:]   
    list_mid = double_nodes(G0,G_sms)

    dict_degree = dict(zip(G0.nodes(),[G0.degree(i,weight = 'weight') for i in G0.nodes()]))  
    dict_degree_sms = dict(zip(G_sms.nodes(),[G_sms.degree(i,weight = 'weight') for i in G_sms.nodes()]))                   

    
    while swapcount < nswap:
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        node1,node2 = random.sample(list_mid,2)
        degree_G1n1, degree_G1n2 = dict_degree[node1],dict_degree[node2]
        degree_G2n1, degree_G2n2 = dict_degree_sms[node1],dict_degree_sms[node2]
        n += 1
        if (degree_G1n1-degree_G1n2)*(degree_G2n1-degree_G2n2) >= 0:
            continue
        else:
            dict_degree = change(dict_degree,node1,node2)
            ia  =node_list_shuffling.index(node1)
            ib  =node_list_shuffling.index(node2)
            node_list_shuffling[ia], node_list_shuffling[ib] = node_list_shuffling[ib], node_list_shuffling[ia]
            swapcount += 1
            print('Valued numbers',swapcount)  
    node_dict = dict(zip(node_list,node_list_shuffling))
    G = nx.relabel_nodes(G, node_dict)
    return G



'''
活跃度相关系数负向（不保证层间度相关系数不变）
'''

def replace_degree_de3(G0,G_sms, nswap=1, max_tries=100):
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")  
    n = 0
    swapcount = 0
    G = copy.deepcopy(G0)   
    node_list=list(G0.nodes()) 
    node_list_shuffling=node_list[:]   
    list_mid = list_mid = double_nodes(G0,G_sms)

    dict_degree = dict(zip(G0.nodes(),[G0.degree(i,weight = 'weight') for i in G0.nodes()]))  
    dict_degree_sms = dict(zip(G_sms.nodes(),[G_sms.degree(i,weight = 'weight') for i in G_sms.nodes()]))                   
               
    while swapcount < nswap:
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        node1,node2 = random.sample(list_mid,2)
        degree_G1n1, degree_G1n2 = dict_degree[node1],dict_degree[node2]
        degree_G2n1, degree_G2n2 = dict_degree_sms[node1],dict_degree_sms[node2]
        n += 1
        if (degree_G1n1-degree_G1n2)*(degree_G2n1-degree_G2n2) <= 0:
            continue
        else:
            dict_degree = change(dict_degree,node1,node2)
            ia  =node_list_shuffling.index(node1)
            ib  =node_list_shuffling.index(node2)
            node_list_shuffling[ia], node_list_shuffling[ib] = node_list_shuffling[ib], node_list_shuffling[ia]
            swapcount += 1
            print('Valued numbers',swapcount)  
    node_dict = dict(zip(node_list,node_list_shuffling))
    G = nx.relabel_nodes(G, node_dict)
    return G
###################################################################################################################################################           


'''
富人俱乐部系数正向

'''

def replace_rich_in1(G0,G_sms, k = 10,k1= 10, nswap=1, max_tries=100):
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")  
    n = 0
    swapcount = 0
    G = copy.deepcopy(G0)   
    node_list=list(G0.nodes()) 
    node_list_shuffling=node_list[:]   # 随机
    list_mid = double_nodes(G,G_sms)  # 取两个网络都存在的节点
    dict_degree = dict(G.degree())   

    while swapcount < nswap:
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        v,u = random.sample(node_list,2)
        if v in list_mid and u in list_mid:  # 判断v,u两个节点是双通节点不
            if G.degree()[v] > k and G.degree()[u] < k and G_sms.degree()[v] < k1 and G_sms.degree()[u] > k1:
                dict_degree = change(dict_degree,v,u)
                ia  =node_list_shuffling.index(v)
                ib  =node_list_shuffling.index(u)
                node_list_shuffling[ia], node_list_shuffling[ib] = node_list_shuffling[ib], node_list_shuffling[ia]
                swapcount += 1
            else:
                continue
        else:
            continue
    print('Valued numbers',swapcount)  
    node_dict = dict(zip(node_list,node_list_shuffling))
    G = nx.relabel_nodes(G, node_dict)
    print(len(G.nodes))
    return G

def replace_rich_in(G0,G_sms, k = 10,k1= 10, nswap=1, max_tries=100):
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")
    n = 0
    swapcount = 0
    G = copy.deepcopy(G0)
    node_list=list(G0.nodes())
    node_list_shuffling=node_list[:]   # 随机
    list_mid = double_nodes(G,G_sms)  # 取两个网络都存在的节点
    # dict_degree = dict(G.degree())
    nodess = []

    while swapcount < nswap:
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        v,u = random.sample(node_list,2)
        # 判断v,u两个节点是双通节点不
        if v not in list_mid and u not in list_mid:
            continue

        # 判断G层网络和G_sms层网络的度是否满足约束条件
        G_vDegree = G.degree(v)
        G_uDegree = G.degree(u)
        if v == u or G_vDegree == G_uDegree:
            continue

        G_sms_vDegree = G_sms.degree(v)
        G_sms_uDegree = G_sms.degree(u)
        if G_sms_vDegree == G_sms_uDegree:
            continue

        if G_vDegree > G_uDegree and G_sms_vDegree > G_sms_uDegree:
            continue
        if G_vDegree < G_uDegree and G_sms_vDegree < G_sms_uDegree:
            continue

        # print(f'G{v}---{G_vDegree}')
        # print(f'G{u}---{G_uDegree}')
        # print(f'G_sms{v}---{G_sms_vDegree}')
        # print(f'G_sms{u}---{G_sms_uDegree}')
        nodess.append(v)
        nodess.append(u)
        # dict_degree = change(dict_degree, v, u)  # 将字典中的v,u交换度数，只交换度数，不交换边好吗？
        ia = node_list_shuffling.index(v)
        ib = node_list_shuffling.index(u)
        node_list_shuffling[ia], node_list_shuffling[ib] = node_list_shuffling[ib], node_list_shuffling[ia]
        swapcount += 1
        # print(f'有效交换 {swapcount}次')

    node_dict = dict(zip(node_list,node_list_shuffling))
    G1 = nx.relabel_nodes(G, node_dict)     # 给图重新标记节点
    # for node in nodess:
    #     print(f'交换后G{node}---{G1.degree(node)}')
    #     print(f'交换和G_sms{node}---{G_sms.degree(node)}')
    # print(len(G.nodes))
    return G1

'''
富人俱乐部系数负向
'''

def replace_rich_de(G0,G_sms,k=10,k1=10, nswap=1, max_tries=100):
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")  
    n = 0
    swapcount = 0
    G = copy.deepcopy(G0)   
    node_list=list(G0.nodes()) 
    node_list_shuffling=node_list[:]   
    list_mid = double_nodes(G,G_sms)
    dict_degree = dict(G.degree())   

              
    while swapcount < nswap:
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        v,u = random.sample(node_list,2)
        if v in list_mid and u in list_mid:
            if G.degree()[v] > k and G.degree()[u] < k and G_sms.degree()[v] > k1 and G_sms.degree()[u] < k1:
                dict_degree = change(dict_degree,v,u)
                ia  =node_list_shuffling.index(v)
                ib  =node_list_shuffling.index(u)
                node_list_shuffling[ia], node_list_shuffling[ib] = node_list_shuffling[ib], node_list_shuffling[ia]
                swapcount += 1
            else:
                continue
        else:
            continue
    print('Valued numbers',swapcount)  
    node_dict = dict(zip(node_list,node_list_shuffling))
    G = nx.relabel_nodes(G, node_dict)
    return G


'''
富人俱乐部系数负向
'''

def replace_rich_random(G0,G_sms,nswap=1, max_tries=100):
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")  
    n = 0
    swapcount = 0
    G = copy.deepcopy(G0)   
    node_list=list(G0.nodes()) 
    node_list_shuffling=node_list[:]   
    list_mid = double_nodes(G,G_sms)
    dict_degree = dict(G.degree())   

              
    while swapcount < nswap:
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        v,u = random.sample(node_list,2)
        if v in list_mid and u in list_mid:
            dict_degree = change(dict_degree,v,u)
            ia  =node_list_shuffling.index(v)
            ib  =node_list_shuffling.index(u)
            node_list_shuffling[ia], node_list_shuffling[ib] = node_list_shuffling[ib], node_list_shuffling[ia]
            swapcount += 1
        else:
            continue
    print('Valued numbers',swapcount)  
    node_dict = dict(zip(node_list,node_list_shuffling))
    G = nx.relabel_nodes(G, node_dict)
    return G






#if __name__ =="__main___":

#    f = open('df_weight.pkl','rb')
#    df_1 = pickle.load(f)
#    f.close()
#    G = nx.from_pandas_dataframe(df_1,'n1','n2',edge_attr='weight',create_using=nx.MultiGraph())








