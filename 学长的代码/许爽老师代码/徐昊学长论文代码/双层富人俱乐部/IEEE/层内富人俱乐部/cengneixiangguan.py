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
    G = nx.from_pandas_dataframe(df,'node1','node2',create_using=nx.Graph())
    return G

'''
平均度
'''
def ave_degree(G):
    all_degree = [i[1] for i in list(G.degree())]
    average_degree = sum(all_degree)/len(all_degree)
    return average_degree


def delown(G_BA_richup):
    for i in list(G_BA_richup.edges()):
        if i[0] == i[1]:
            G_BA_richup.remove_edge(i[0], i[1])
    return G_BA_richup



df_call = pd.read_csv('new_IEEE118.csv')
df_sms = pd.read_csv('new_Internet118.csv')
#
#
G_call = creat_G(df_call)
G_sms = creat_G(df_sms)
##
G_call = delown(G_call)
G_sms = delown(G_sms)
G_call_richup = unm.rich_club_create(G_call,k = 4,nswap=20000000, max_tries=40000000, connected=1)
G_call_richup = delown(G_call_richup)
G_call_richdown = unm.rich_club_break(G_call,k = 4,nswap=20000000, max_tries=40000000, connected=1)
G_call_richdown = delown(G_call_richdown)
G_sms_richup = unm.rich_club_create(G_sms,k = 11,nswap=20000000, max_tries=40000000, connected=1)
G_sms_richup = delown(G_sms_richup)
G_sms_richdown = unm.rich_club_break(G_sms,k = 11,nswap=20000000, max_tries=40000000, connected=1)
G_sms_richdown = delown(G_sms_richdown)
##
rc_call = nx.rich_club_coefficient(G_call,normalized = False)[4]
rc_call_richup = nx.rich_club_coefficient(G_call_richup,normalized = False)[4]
rc_call_richdown = nx.rich_club_coefficient(G_call_richdown,normalized = False)[4]
rc_sms = nx.rich_club_coefficient(G_sms,normalized = False)[11]
rc_sms_richup = nx.rich_club_coefficient(G_sms_richup,normalized = False)[11]
rc_sms_richdown = nx.rich_club_coefficient(G_sms_richdown,normalized = False)[11]

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


df_cas = mergedf(df_call,df_sms)
df_cas.to_csv('IaI.csv',index = False)
#



df_call_richup = G2df(G_call_richup)
df_call_richup.to_csv('IEEE_richup.csv',index = False)
#df_call_richup['network'] = 0
#df_cas_callrichup = mergedf(df_call_richup,df_sms)
#df_cas_callrichup.to_csv('IaI_IEEErichup.csv',index = False)


df_call_richdown = G2df(G_call_richdown)
df_call_richdown.to_csv('IEEE_richdown-0.0.csv',index = False)
#df_call_richdown['network'] = 0
#df_cas_callrichdown = mergedf(df_call_richdown,df_sms)
#df_cas_callrichdown.to_csv('IaI_IEEErichdown.csv',index = False)

df_sms_richup = G2df(G_sms_richup)
df_sms_richup.to_csv('Internet_richup.csv',index = False)
#df_sms_richup['network'] = 0
#df_cas_smsrichup = mergedf(df_sms_richup,df_call)
#df_cas_smsrichup.to_csv('IaI_smsrichup.csv',index = False)

df_sms_richdown = G2df(G_sms_richdown)
df_sms_richdown.to_csv('Internet_richdown.csv',index = False)
#df_sms_richdown['network'] = 0
#df_cas_smsrichdown = mergedf(df_sms_richdown,df_call)
#df_cas_smsrichdown.to_csv('IaI_Internetrichdown.csv',index = False)

####
#
#
#G_cas = creat_G(df_cas)
#G_cas = delown(G_cas)
#G_cas_callrichup = creat_G(df_cas_callrichup)
#G_cas_callrichup = delown(G_cas_callrichup)
#G_cas_callrichdown = creat_G(df_cas_callrichdown)
#G_cas_callrichdown = delown(G_cas_callrichdown)
#G_cas_smsrichup = creat_G(df_cas_smsrichup)
#G_cas_smsrichup = delown(G_cas_smsrichup)
#G_cas_smsrichdown = creat_G(df_cas_smsrichdown)
#G_cas_smsrichdown = delown(G_cas_smsrichdown)
#
#hubs = [e for e in G_call.nodes() if G_call.degree(e)>4]#21
#hubs = [e for e in G_cas.nodes() if G_cas.degree(e)>14] #23
#hubs = [e for e in G_cas_callrichdown.nodes() if G_cas_callrichdown.degree(e)>14] #24
#hubs = [e for e in G_cas_callrichup.nodes() if G_cas_callrichup.degree(e)>14] #22
#hubs = [e for e in G_cas_smsrichdown.nodes() if G_cas_smsrichdown.degree(e)>14] #23
#hubs = [e for e in G_cas_smsrichup.nodes() if G_cas_smsrichup.degree(e)>14] #24
## 
#rc_cas = nx.rich_club_coefficient(G_cas,normalized = False)[14] #富节点1550
#rc_cas_callrichup = nx.rich_club_coefficient(G_cas_callrichup,normalized = False)[14]#富节点1572
#rc_cas_callrichdown = nx.rich_club_coefficient(G_cas_callrichdown,normalized = False)[14]#富节点1560
#rc_cas_smsrichup = nx.rich_club_coefficient(G_cas_smsrichup,normalized = False)[14]#富节点1560
#rc_cas_smsrichdown = nx.rich_club_coefficient(G_cas_smsrichdown,normalized = False)[14] #富节点1560

