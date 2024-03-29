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
import node_replace as nr






def creat_weighted_G(df):
    G = nx.from_pandas_edgelist(df,'node1','node2',edge_attr='weight',create_using=nx.MultiDiGraph())
    return G
def creat_G(df):
    G = nx.from_pandas_edgelist(df,'node1','node2',create_using=nx.MultiGraph())
    return G
'''
双通节点
'''
def double_nodes(G_call,G_sms):
    return list(set(G_call.nodes()).intersection(set(G_sms.nodes())))



'''
双通节点比例
'''
def double_nodes_ratio(G_call,G_sms):
    list_allnodes = [val for val in G_call.nodes() if val in G_sms.nodes()]
    len_call_double = len([i for i in G_call.nodes() if i in list_allnodes])
    len_sms_double = len([i for i in G_sms.nodes() if i in list_allnodes])
    return len_call_double/len(G_call.nodes()),len_sms_double/len(G_sms.nodes())
'''
平均度
'''
def ave_degree(G):
    all_degree = [i[1] for i in list(G.degree())]
    average_degree = sum(all_degree)/len(all_degree)
    return average_degree

'''
网络中每个节点的强度，即每个节点的权重和
'''
def strength(G_call):
    list_s = [G_call.degree(i,weight = 'weight') for i in list(G_call.nodes())]
    dict_s = dict(zip(list(G_call.nodes()),list_s))
    return dict_s


def ave_strength(G_call):
    list_s = [G_call.degree(i,weight = 'weight') for i in list(G_call.nodes())]
    a = sum(list_s)/len(list_s)
    return a

'''
两个网络的皮尔森层间度相关系数
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
    return sum_ab,math.sqrt(sum_a1),math.sqrt(sum_b1),sum_ab/math.sqrt(sum_a1)/math.sqrt(sum_b1)
        


'''
两个网络的皮尔森层间通信活跃度相关系数
'''

def pearson_active(G1,G2):     
    list1 = strength(G1)
    list2 = strength(G2)
    list_mid = double_nodes(G1,G2)
    d1 = ave_strength(G1)
    d2 = ave_strength(G2)
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
    return sum_ab/math.sqrt(sum_a1)/math.sqrt(sum_b1)





df_call = pd.read_csv('call.csv')
df_sms = pd.read_csv('sms.csv')
del(df_call['weight'])
del(df_sms['weight'])
df_call['network'] = 0
df_sms['network'] = 1
#
#
G_call = creat_G(df_call)
G_sms = creat_G(df_sms)


G_call_pdup = nr.replace_degree_in2(G_call,G_sms,100000,2000000)
G_call_pddown = nr.replace_degree_de2(G_call,G_sms,100000,2000000)
G_call_pdrandom = nr.replace_degree_random(G_call,G_sms,2000000,2000000)


pd_origin = pearson_degree(G_call,G_sms)[3]
pd_up = pearson_degree(G_call_pdup,G_sms)[3]
pd_down = pearson_degree(G_call_pddown,G_sms)[3]
pd_random = pearson_degree(G_call_pdrandom,G_sms)[3]



'''
合并双层网络并加权处理
'''
def mergedf(df1,df2):
    df3 = df1.append(df2)
#    df4 = df3.groupby(['node1', 'node2'])['weight'].sum().reset_index()
    return df3

'''
将无权网络转为dataframe
'''

def G2df(G): 
    
    list_n1,list_n2 = [],[]
    for i in range(len(G.edges())):
        list_n1.append(list(G.edges())[i][0])
        list_n2.append(list(G.edges())[i][1])
    df = pd.DataFrame(({'node1':list_n1,'node2':list_n2}))
    return df

'''
将加权网络转为dataframe
'''
def G2G2df(G): 
    dict_a = nx.get_edge_attributes(G,'weight')
    b = list(dict_a.keys())
    list_t = list(dict_a.values())
    list_n1,list_n2 = [],[]
    for i in b:
        list_n1.append(list(i)[0])
        list_n2.append(list(i)[1])
    df = pd.DataFrame(({'node1':list_n1,'node2':list_n2,'weight':list_t}))
    return df


#
df_call_pdup = G2df(G_call_pdup)
df_call_pdup.to_csv('call_pdup.csv',index = False)
df_call_pdup['network'] = 0
df_cas_callpdup = mergedf(df_call_pdup,df_sms)
df_cas_callpdup.to_csv('cas_callpdup.csv',index = False)
##
df_call_pddown = G2df(G_call_pddown)
df_call_pddown.to_csv('call_pddown.csv',index = False)
df_call_pddown['network'] = 0
df_cas_callpddown = mergedf(df_call_pddown,df_sms)
df_cas_callpddown.to_csv('cas_callpddown.csv',index = False)


df_call_pdrandom = G2df(G_call_pdrandom)
df_call_pdrandom.to_csv('call_pdrandom.csv',index = False)
df_call_pdrandom['network'] = 0
df_cas_callpdrandom = mergedf(df_call_pdrandom,df_sms)
df_cas_callpdrandom.to_csv('cas_callpdrandom.csv',index = False)




