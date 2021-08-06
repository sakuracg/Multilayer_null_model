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
import time






def creat_weighted_G(df):
    G = nx.from_pandas_edgelist(df,'node1','node2',edge_attr='weight',create_using=nx.MultiDiGraph())
    return G
def creat_G(df):
    G = nx.from_pandas_edgelist(df,'node1','node2',create_using=nx.Graph())
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

def rc(G1,G2,m,n):
    hubs1 = [e for e in G1.nodes() if G1.degree(e)>m]
    hubs2 = [e for e in G2.nodes() if G2.degree(e)>n]
    hubs1.sort()
    hubs2.sort()
    bj = list(set(hubs1).union(set(hubs2)))
    jj = list(set(hubs1).intersection(set(hubs2)))
    return len(jj)/len(bj)



df_BA = pd.read_csv('BA.csv')
df_ws = pd.read_csv('ws.csv')
df_BA['network'] = 0
df_ws['network'] = 1
#df_BA_richdown= pd.read_csv('BA_richdown.csv')
#df_BA_richup = pd.read_csv('BA_richup.csv')
#df_BA_richrandom = pd.read_csv('BA_richrandom.csv')
#
#
G_BA = creat_G(df_BA)
G_ws = creat_G(df_ws)
#G_BA_richup = creat_G(df_BA_richup)
#G_BA_richdown = creat_G(df_BA_richdown)
#G_BA_richrandom = creat_G(df_BA_richrandom)
# richclub = rc(G_BA,G_ws,7,6)
# print(richclub)

start_time = int(round(time.time() * 1000))
G_BA_richup = nr.replace_rich_in(G_BA,G_ws, k=7, k1=6, nswap=400, max_tries=2000000)
end_time = int(round(time.time() * 1000))
print(f"学长算法执行时间{end_time - start_time}")
rc_BAup = rc(G_BA_richup,G_ws,7,6)
print(rc_BAup)
# pass
# G_BA_richdown = nr.replace_rich_de(G_BA,G_ws, k = 7,k1= 6, nswap=15000, max_tries=2000000)
# rc_BAdown = rc(G_BA_richdown,G_ws,7,6)
#
# G_BA_richrandom = nr.replace_rich_random(G_BA,G_ws,nswap=15000, max_tries=2000000)
# rc_BArandom = rc(G_BA_richrandom,G_ws,7,6)

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

# df_BA_richup = G2df(G_BA_richup)
# df_BA_richup.to_csv('BA_richup.csv',index = False)
#
#
# df_BA_richup = pd.read_csv('BA_richup.csv')
# df_ws1 = pd.read_csv('ws.csv')
# df_BA_richup['network'] = 0
# df_ws1['network'] = 1
# #df_BA_richdown= pd.read_csv('BA_richdown.csv')
# #df_BA_richup = pd.read_csv('BA_richup.csv')
# #df_BA_richrandom = pd.read_csv('BA_richrandom.csv')
# #
# #
# G_BA = creat_G(df_BA_richup)
# G_ws = creat_G(df_ws1)
# richclub = rc(G_BA,G_ws,7,6)
# print(richclub)
#
#
# df_BA_richup['network'] = 0
# df_Baw_BArichup = mergedf(df_BA_richup,df_ws)
# df_Baw_BArichup.to_csv('Baw_richup.csv',index = False)
#
# df_BA_richdown = G2df(G_BA_richdown)
# df_BA_richdown.to_csv('BA_richdown.csv',index = False)
# df_BA_richdown['network'] = 0
# df_Baw_BArichdown = mergedf(df_BA_richdown,df_ws)
# df_Baw_BArichdown.to_csv('Baw_richdown.csv',index = False)
#
# df_BA_richrandom = G2df(G_BA_richrandom)
# df_BA_richrandom.to_csv('BA_richrandom.csv',index = False)
# df_BA_richrandom['network'] = 0
# df_Baw_BArichrandom = mergedf(df_BA_richdown,df_ws)
# df_Baw_BArichrandom.to_csv('Baw_richrandom.csv',index = False)
#
