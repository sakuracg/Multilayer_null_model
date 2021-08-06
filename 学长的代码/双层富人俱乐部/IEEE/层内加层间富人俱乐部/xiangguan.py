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

def rc(G1,G2,m,n):
    hubs1 = [e for e in G1.nodes() if G1.degree(e)>m]
    hubs2 = [e for e in G2.nodes() if G2.degree(e)>n]
    bj = list(set(hubs1).union(set(hubs2)))
    jj = list(set(hubs1).intersection(set(hubs2)))
    return len(jj)/len(bj)


df_BArichup = pd.read_csv('IEEE_richup.csv')
df_BArichdown = pd.read_csv('IEEE_richdown.csv')
df_wsrichup = pd.read_csv('Internet_richup.csv')
df_wsrichdown = pd.read_csv('Internet_richdown.csv')

#
#
G_BA_pdup = creat_G(df_BArichup)
G_BA_pddown = creat_G(df_BArichdown)
G_ws_pdup = creat_G(df_wsrichup)
G_ws_pddown = creat_G(df_wsrichdown)






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
'''
twoup（分开运行）
'''
hubs1 = [e for e in G_BA_pdup.nodes() if G_BA_pdup.degree(e)>4] #全部富节点19
hubs2 = [e for e in G_BA_pddown.nodes() if G_BA_pddown.degree(e)>4] #全部富节点21
hubs3 = [e for e in G_ws_pdup.nodes() if G_ws_pdup.degree(e)>11] #全部富节点23
hubs4 = [e for e in G_ws_pddown.nodes() if G_ws_pddown.degree(e)>11] #全部富节点24

G_BA_pdup_interup = nr.replace_rich_in(G_BA_pdup,G_ws_pdup,k=4,k1=11, nswap=2000000, max_tries=1000000000)
G_BA_pdup_interdown = nr.replace_rich_de(G_BA_pdup,G_ws_pdup,k=4,k1=11, nswap=2000000, max_tries=100000000)
G_BA_pdup_interrandom = nr.replace_rich_random(G_BA_pdup,G_ws_pdup,2000000,20000000)
#
rc_twoup = rc(G_BA_pdup,G_ws_pdup,4,11)
rc_BA_richup_interup = rc(G_BA_pdup_interup,G_ws_pdup,4,11)
rc_BA_richup_interdown = rc(G_BA_pdup_interdown,G_ws_pdup,4,11)
rc_BA_richup_interrandom = rc(G_BA_pdup_interrandom,G_ws_pdup,4,11)



##
df_BA_pdup = G2df(G_BA_pdup_interup)
df_BA_pdup.to_csv('IEEE_richup_interup.csv',index = False)

##
df_BA_pddown = G2df(G_BA_pdup_interdown)         
df_BA_pddown.to_csv('IEEE_richup_interdown.csv',index = False)           


df_BA_pdrandom = G2df(G_BA_pdup_interrandom)
df_BA_pdrandom.to_csv('IEEE_richup_interrandom.csv',index = False)



'''
twodown（分开运行）
'''
#G_BA_pddown_interup = nr.replace_rich_in(G_BA_pddown,G_ws_pddown,k=4,k1=11, nswap=2000000, max_tries=100000000)
#G_BA_pddown_interdown = nr.replace_rich_de(G_BA_pddown,G_ws_pddown,k=4,k1=11, nswap=2000000, max_tries=100000000)
#G_BA_pddown_interrandom = nr.replace_rich_random(G_BA_pddown,G_ws_pddown,2000000,2000000)
#
#rc_twodown = rc(G_BA_pddown,G_ws_pddown,4,11)
#rc_BA_richdown_interup = rc(G_BA_pddown_interup,G_ws_pddown,4,11)
#rc_BA_richdown_interdown = rc(G_BA_pddown_interdown,G_ws_pddown,4,11)
#rc_BA_richdown_interrandom = rc(G_BA_pddown_interrandom,G_ws_pddown,4,11)
#
#
#
###0
#df_BA_pdup = G2df(G_BA_pddown_interup)
#df_BA_pdup.to_csv('IEEE_richdown_interup.csv',index = False)
###
#df_BA_pddown = G2df(G_BA_pddown_interdown)             
#df_BA_pddown.to_csv('IEEE_richdown_interdown.csv',index = False)   
#
#
#df_BA_pdrandom = G2df(G_BA_pddown_interrandom)
#df_BA_pdrandom.to_csv('IEEE_richdown_interrandom.csv',index = False)



#########################################################################################################################################

#df_BApdup = pd.read_csv('BA_pdup.csv')
#df_BApdup['network'] = 0
#df_BApddown = pd.read_csv('BA_pddown.csv')
#df_BApddown['network'] = 0
#df_ws_pdup = pd.read_csv('ws_pdup.csv')
#df_ws_pdup['network'] = 1
#df_ws_pddown = pd.read_csv('ws_pddown.csv')
#df_ws_pddown['network'] = 1
#
#df_BA_pdup_interup = pd.read_csv('BA_pdup_interup.csv')
#df_BA_pdup_interup['network'] = 0
#df_Baw_BApdup_interup = mergedf(df_BA_pdup_interup,df_ws_pdup)
#df_Baw_BApdup_interup.to_csv('Baw_pdup_interup.csv',index = False)
#
#df_BA_pdup_interdown = pd.read_csv('BA_pdup_interdown.csv')
#df_BA_pdup_interdown['network'] = 0
#df_Baw_BApdup_interdown = mergedf(df_BA_pdup_interdown,df_ws_pdup)
#df_Baw_BApdup_interdown.to_csv('Baw_pdup_interdown.csv',index = False)
#
#df_BA_pdup_interrandom = pd.read_csv('BA_pdup_interrandom.csv')
#df_BA_pdup_interrandom['network'] = 0
#df_Baw_BApdup_interrandom = mergedf(df_BA_pdup_interrandom,df_ws_pdup)
#df_Baw_BApdup_interrandom.to_csv('Baw_pdup_interrandom.csv',index = False)
#
#df_BA_pddown_interup = pd.read_csv('BA_pddown_interup.csv')
#df_BA_pddown_interup['network'] = 0
#df_Baw_BApddown_interup = mergedf(df_BA_pddown_interup,df_ws_pddown)
#df_Baw_BApddown_interup.to_csv('Baw_pddown_interup.csv',index = False)
#
#df_BA_pddown_interdown = pd.read_csv('BA_pddown_interdown.csv')
#df_BA_pddown_interdown['network'] = 0
#df_Baw_BApddown_interdown = mergedf(df_BA_pddown_interdown,df_ws_pddown)
#df_Baw_BApddown_interdown.to_csv('Baw_pddown_interdown.csv',index = False)
#
#df_BA_pddown_interrandom = pd.read_csv('BA_pddown_interrandom.csv')
#df_BA_pddown_interrandom['network'] = 0
#df_Baw_BApddown_interrandom = mergedf(df_BA_pddown_interrandom,df_ws_pddown)
#df_Baw_BApddown_interrandom.to_csv('Baw_pddown_interrandom.csv',index = False)
#
#
#
#
#df_Baw_twoup = mergedf(df_BApdup,df_ws_pdup)
#df_Baw_twoup.to_csv('Baw_tworichup.csv',index = False)
#
#df_Baw_twodown = mergedf(df_BApddown,df_ws_pddown)
#df_Baw_twodown.to_csv('Baw_tworichdown.csv',index = False)


       





