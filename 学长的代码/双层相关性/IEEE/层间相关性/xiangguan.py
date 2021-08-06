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



def cengjian():
    pd1 = pearson_degree(G_call,G_sms)[3]
    pd_gai = pearson_degree(G_call_gai1,G_sms)[3]
    pc = pearson_active(G_call,G_sms)
    pc_gai = pearson_active(G_call_gai1,G_sms)
    print('电话网络与短信网络的层间度相关系数,电话网络（改）与短信网络的层间度相关系数 :\n',pd,pd_gai)    
    print('电话网络与短信网络的层间活跃度相关系数,电话网络（改）与短信网络的层间活跃度相关系数 :\n',pc,pc_gai)  

df_BA = pd.read_csv('new_IEEE118.csv')
df_ws = pd.read_csv('new_Internet118.csv')
#
#
G_BA = creat_G(df_BA)
G_ws = creat_G(df_ws)

G_BA_pdup = nr.replace_degree_in2(G_BA,G_ws,100000,2000000)
G_BA_pddown = nr.replace_degree_de2(G_BA,G_ws,100000,2000000)
G_BA_pdrandom = nr.replace_degree_random(G_BA,G_ws,2000000,2000000)

pd_origin = pearson_degree(G_BA,G_ws)[3]
pd_up = pearson_degree(G_BA_pdup,G_ws)[3]
pd_down = pearson_degree(G_BA_pddown,G_ws)[3]
pd_random = pearson_degree(G_BA_pdrandom,G_ws)[3]



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






#df_BA = pd.read_csv('BA.csv')
#df_BA['network'] = 0
#df_ws = pd.read_csv('ws.csv')
#df_ws['network'] = 1
#df_BA_pdup = pd.read_csv('BA_pdup.csv')
#df_BA_pdup['network'] = 0
#df_BA_pddown = pd.read_csv('BA_pddown.csv')
#df_BA_pddown['network'] = 0
#df_BA_pdrandom = pd.read_csv('BA_pdrandom.csv')
#df_BA_pdrandom['network'] = 0
#            
#
#G_BA = creat_G(df_BA)
#G_ws = creat_G(df_ws)
#G_BA_pdup = creat_G(df_BA_pdup)
#G_BA_pddown = creat_G(df_BA_pddown)
#G_BA_pdrandom = creat_G(df_BA_pdrandom)        
#
#
#pd_origin = pearson_degree(G_BA,G_ws)[3]
#pd_up = pearson_degree(G_BA_pdup,G_ws)[3]
#pd_down = pearson_degree(G_BA_pddown,G_ws)[3]
#pd_random = pearson_degree(G_BA_pdrandom,G_ws)[3]
#
#
##df_Baw = mergedf(df_BA,df_ws)
##df_Baw.to_csv('Baw.csv',index = False)
##
##df_Baw_BApdup = mergedf(df_BA_pdup,df_ws)
##df_Baw_BApdup.to_csv('Baw_pdup.csv',index = False)
##
##df_Baw_BApddown = mergedf(df_BA_pddown,df_ws)
##df_Baw_BApddown.to_csv('Baw_pddown.csv',index = False)
##
##df_Baw_BApdrandom = mergedf(df_BA_pdrandom,df_ws)
##df_Baw_BApdrandom.to_csv('Baw_pdrandom.csv',index = False)

