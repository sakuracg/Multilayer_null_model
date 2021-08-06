# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:11:47 2020

@author: Dell
"""

import pandas as pd
import networkx as nx
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import math
import unweight_null_model as unm






def creat_G(df):
    G = nx.from_pandas_edgelist(df,'node1','node2',create_using=nx.Graph())
    return G



'''
双通节点
'''
def double_nodes(G_call,G_sms):
    return list(set(G_call.nodes()).intersection(set(G_sms.nodes())))

'''
平均度
'''
def ave_degree(G):
    all_degree = [i[1] for i in list(G.degree())]
    average_degree = sum(all_degree)/len(all_degree)
    return average_degree


'''
联通性  全联通为True
'''
def connect(G):
    return nx.is_connected(G)

'''
层内度相关系数
'''
def dac(G):
    return nx.degree_assortativity_coefficient(G)

'''
两个网络的层间度相关系数
'''

def pearson_degree(G1,G2):     
    list1 = G1.degree()
    list2 = G2.degree()
    list_mid = double_nodes(G1,G2)
    d1 = ave_degree(G1)
    d2 = ave_degree(G2)
#     N=len(list1) 
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
    return sum_ab,math.sqrt(sum_a1),math.sqrt(sum_b1),sum_ab/math.sqrt(sum_a1)/math.sqrt(sum_b1)

'''
层内富人俱乐部系数(取度值大于n的节点为富节点)
'''
def rcc(G,n):
    return nx.rich_club_coefficient(G,normalized = False)[n]
'''
层间富人俱乐部系数(G1取度值大于m的节点为富节点,G2取度值大于n的节点为富节点)
'''
def rc(G1,G2,m,n):
    hubs1 = [e for e in G1.nodes() if G1.degree(e)>m]
    hubs2 = [e for e in G2.nodes() if G2.degree(e)>n]
    bj = list(set(hubs1).union(set(hubs2)))
    jj = list(set(hubs1).intersection(set(hubs2)))
    return len(jj)/len(bj)
'''
富节点个数
'''
def rcc_num(G,n):
    return len([G for e in G.nodes() if G.degree(e)>n])

"""
平均路径长度
"""
def average_shorted_pathLength(G):
    return nx.average_shortest_path_length(G)


##
##

if __name__ == "__main__":
    df_BA = pd.read_csv('./BA.csv')

    # df_ws = pd.read_csv('D:\python\multi_layer_networks\task\ws.csv')

    # G_BA = creat_G(df_BA)
    # G_ws = creat_G(df_ws)
    # print(connect(G_BA))



