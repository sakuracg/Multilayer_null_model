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
    G = nx.from_pandas_dataframe(df,'node1','node2',edge_attr='weight',create_using=nx.MultiDiGraph())
    return G
def creat_G(df):
    G = nx.from_pandas_dataframe(df,'node1','node2',create_using=nx.Graph())
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
    bj = list(set(hubs1).union(set(hubs2)))
    jj = list(set(hubs1).intersection(set(hubs2)))
    return len(jj)/len(bj)




df_call = pd.read_csv('new_IEEE118.csv')
df_sms = pd.read_csv('new_Internet118.csv')
#
#
G_call = creat_G(df_call)
G_sms = creat_G(df_sms)
richclub = rc(G_call,G_sms,4,11)
#
G_call_richup = nr.replace_rich_in(G_call,G_sms, k = 4,k1= 11, nswap=2000000, max_tries=100000000)
rc_callup = rc(G_call_richup,G_sms,4,11)

G_call_richdown = nr.replace_rich_de(G_call,G_sms, k = 4,k1= 11, nswap=2000000, max_tries=10000000)
rc_calldown = rc(G_call_richdown,G_sms,4,11)

G_call_richrandom = nr.replace_rich_random(G_call,G_sms,nswap=2000000, max_tries=10000000)
rc_callrandom = rc(G_call_richrandom,G_sms,4,11)
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




df_call_richup = G2df(G_call_richup)
df_call_richup.to_csv('IEEE_richup.csv',index = False)

#
df_call_richdown = G2df(G_call_richdown)
df_call_richdown.to_csv('IEEE_richdown-0.0.csv',index = False)


#
df_call_richrandom = G2df(G_call_richrandom)
df_call_richrandom.to_csv('IEEE_richrandom.csv',index = False)

#
