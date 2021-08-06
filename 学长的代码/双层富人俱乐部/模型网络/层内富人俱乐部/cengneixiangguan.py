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
平均度
'''
def ave_degree(G):
    all_degree = [i[1] for i in list(G.degree())]
    average_degree = sum(all_degree)/len(all_degree)
    return average_degree


def delown(G_BA_richup):
    ll = []
    for i in list(G_BA_richup.edges()):
        if i[0] == i[1]:
            ll.append(i)
            G_BA_richup.remove_edge(i[0], i[1])
    return G_BA_richup



df_BA = pd.read_csv('BA.csv')
df_ws = pd.read_csv('ws.csv')
df_BA['network'] = 0
df_ws['network'] = 1
#
#
G_BA = creat_G(df_BA)
G_ws = creat_G(df_ws)
#

G_BA_richup, val = unm.rich_club_create(G_BA,k = 7,nswap=1, max_tries=40000000, connected=1)
G_BA_richup = delown(G_BA_richup)
G_BA_richdown, val = unm.rich_club_break(G_BA,k=7,nswap=300, max_tries=40000000, connected=1)
G_BA_richdown = delown(G_BA_richdown)
G_ws_richup = unm.rich_club_create(G_ws,k = 6,nswap=100, max_tries=40000000, connected=1)
G_ws_richup = delown(G_ws_richup)
G_ws_richdown = unm.rich_club_break(G_ws,k = 6,nswap=100, max_tries=40000000, connected=1)
G_ws_richdown = delown(G_ws_richdown)
#
rc_BA = nx.rich_club_coefficient(G_BA,normalized = False)[7]
rc_BA_richup = nx.rich_club_coefficient(G_BA_richup,normalized = False)
rc_BA_richup = nx.rich_club_coefficient(G_BA_richup,normalized = False)[7]
rc_BA_richdown = nx.rich_club_coefficient(G_BA_richdown,normalized = False)[7]
rc_ws = nx.rich_club_coefficient(G_ws,normalized = False)[6]
rc_ws_richup = nx.rich_club_coefficient(G_ws_richup,normalized = False)[6]
rc_ws_richdown = nx.rich_club_coefficient(G_ws_richdown,normalized = False)[6]

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



df_Baw = mergedf(df_BA,df_ws)
df_Baw.to_csv('Baw.csv',index = False)



df_BA_richup = G2df(G_BA_richup)
df_BA_richup.to_csv('BA_richup.csv',index = False)
df_BA_richup['network'] = 0
df_Baw_BArichup = mergedf(df_BA_richup,df_ws)
df_Baw_BArichup.to_csv('Baw_BArichup.csv',index = False)
###
df_BA_richdown = G2df(G_BA_richdown)
df_BA_richdown.to_csv('BA_richdown.csv',index = False)
df_BA_richdown['network'] = 0
df_Baw_BArichdown = mergedf(df_BA_richdown,df_ws)
df_Baw_BArichdown.to_csv('Baw_BArichdown.csv',index = False)
#
df_ws_richup = G2df(G_ws_richup)
df_ws_richup.to_csv('ws_richup.csv',index = False)
df_ws_richup['network'] = 1
df_Baw_wsrichup = mergedf(df_ws_richup,df_BA)
df_Baw_wsrichup.to_csv('Baw_wsrichup.csv',index = False)
###
df_ws_richdown = G2df(G_ws_richdown)
df_ws_richdown.to_csv('ws_richdown.csv',index = False)
df_ws_richdown['network'] = 1
df_Baw_wsrichdown = mergedf(df_ws_richdown,df_BA)
df_Baw_wsrichdown.to_csv('Baw_wsrichdown.csv',index = False)


G_Baw = creat_G(df_Baw)
G_Baw_BApdup = creat_G(df_Baw_BArichup)
G_Baw_BApddown = creat_G(df_Baw_BArichdown)
G_Baw_wspdup = creat_G(df_Baw_wsrichup)
G_Baw_wspddown = creat_G(df_Baw_wsrichdown)

hubs = len([G_Baw_wspddown for e in G_Baw_wspddown.nodes() if G_Baw_wspddown.degree(e)>13])    #全部富节点

          
          
rc_Baw = nx.rich_club_coefficient(G_Baw,normalized = False)[13]
rc_Baw_BArichup = nx.rich_club_coefficient(G_Baw_BApdup,normalized = False)[13]
rc_Baw_BArichdown = nx.rich_club_coefficient(G_Baw_BApddown,normalized = False)[13]
rc_Baw_wsrichup = nx.rich_club_coefficient(G_Baw_wspdup,normalized = False)[13]
rc_Baw_wsrichdown = nx.rich_club_coefficient(G_Baw_wspddown,normalized = False)[13]

#################################################################################
df_BA = pd.read_csv('BA.csv')
df_ws = pd.read_csv('ws.csv')
df_BA_richdown = pd.read_csv('BA_richdown.csv')
df_ws_richdown = pd.read_csv('ws_richdown.csv')
df_BA_richup = pd.read_csv('BA_richup.csv')
df_ws_richup = pd.read_csv('ws_richup.csv')
df_Baw = pd.read_csv('Baw.csv')
df_Baw_BArichup = pd.read_csv('Baw_BArichup.csv')
df_Baw_BArichdown = pd.read_csv('Baw_BArichdown.csv')
df_Baw_wsrichup = pd.read_csv('Baw_wsrichup.csv')
df_Baw_wsrichdown = pd.read_csv('Baw_wsrichdown.csv')


G_BA = creat_G(df_BA)
G_ws = creat_G(df_ws)
G_BA_richup = creat_G(df_BA_richup)
G_BA_richdown = creat_G(df_BA_richdown)
G_ws_richup = creat_G(df_ws_richup)
G_ws_richdown = creat_G(df_ws_richdown)
G_Baw = creat_G(df_Baw)
G_Baw_BApdup = creat_G(df_Baw_BArichup)
G_Baw_BApddown = creat_G(df_Baw_BArichdown)
G_Baw_wspdup = creat_G(df_Baw_wsrichup)
G_Baw_wspddown = creat_G(df_Baw_wsrichdown)

hubs = len([G_Baw_wspddown for e in G_Baw_wspddown.nodes() if G_Baw_wspddown.degree(e)>13])    #全部富节点

          
          
rc_Baw = nx.rich_club_coefficient(G_Baw,normalized = False)[13]
rc_Baw_BArichup = nx.rich_club_coefficient(G_Baw_BApdup,normalized = False)[13]
rc_Baw_BArichdown = nx.rich_club_coefficient(G_Baw_BApddown,normalized = False)[13]
rc_Baw_wsrichup = nx.rich_club_coefficient(G_Baw_wspdup,normalized = False)[13]
rc_Baw_wsrichdown = nx.rich_club_coefficient(G_Baw_wspddown,normalized = False)[13]
rc_BA = nx.rich_club_coefficient(G_BA,normalized = False)[7]
rc_BA_richup = nx.rich_club_coefficient(G_BA_richup,normalized = False)[7]
rc_BA_richdown = nx.rich_club_coefficient(G_BA_richdown,normalized = False)[7]
rc_ws = nx.rich_club_coefficient(G_ws,normalized = False)[6]
rc_ws_richup = nx.rich_club_coefficient(G_ws_richup,normalized = False)[6]
rc_ws_richdown = nx.rich_club_coefficient(G_ws_richdown,normalized = False)[6]



