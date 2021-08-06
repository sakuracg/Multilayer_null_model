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






#df_BA = pd.read_table('new_IEEE118.txt',sep = ' ',names = ['node1','node2'])
#df_ws = pd.read_table('new_Internet118.txt',sep = ' ',names = ['node1','node2'])
#df_BA.to_csv('new_IEEE118.csv',index = False)
#df_ws.to_csv('new_Internet118.csv',index = False)

df_BA = pd.read_csv('new_IEEE118.csv')
df_ws = pd.read_csv('new_Internet118.csv')
##
##
G_BA = creat_G(df_BA)
G_ws = creat_G(df_ws)
#
G_BA_pdup = unm.assort_mixing(G_BA,nswap=500000, max_tries=500000, connected=1)
G_BA_pddown = unm.disassort_mixing(G_BA,nswap=500000, max_tries=500000, connected=1)
G_ws_pdup = unm.assort_mixing(G_ws,nswap=500000, max_tries=5000000, connected=1)
G_ws_pddown = unm.disassort_mixing(G_ws,nswap=500000, max_tries=500000, connected=1)

dp_BA = nx.degree_assortativity_coefficient(G_BA)
dp_BA_pdup = nx.degree_assortativity_coefficient(G_BA_pdup)
dp_BA_pddown = nx.degree_assortativity_coefficient(G_BA_pddown)
dp_ws = nx.degree_assortativity_coefficient(G_ws)
dp_ws_pdup = nx.degree_assortativity_coefficient(G_ws_pdup)
dp_ws_pddown = nx.degree_assortativity_coefficient(G_ws_pddown)


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




df_BA = G2df(G_BA)
df_ws = G2df(G_ws)
df_BA_pdup = G2df(G_BA_pdup)
df_ws_pdup = G2df(G_ws_pdup)
df_BA_pddown = G2df(G_BA_pddown)
df_ws_pddown = G2df(G_ws_pddown)
df_BA_pdup.to_csv('IEEE_pdup.csv',index = False)
df_ws_pdup.to_csv('Internet_pdup.csv',index = False)
df_BA_pddown.to_csv('IEEE_pddown.csv',index = False)
df_ws_pddown.to_csv('Internet_pddown.csv',index = False)



df_Baw = mergedf(df_BA,df_ws)
df_Baw.to_csv('IaI.csv',index = False)
#
#
df_Baw_BApdup = mergedf(df_BA_pdup,df_ws)
df_Baw_BApdup.to_csv('IaI_IEEEpdup.csv',index = False)
#
#
df_Baw_BApddown = mergedf(df_BA_pddown,df_ws)
df_Baw_BApddown.to_csv('IaI_IEEEpddown.csv',index = False)
#
#
df_Baw_wspdup = mergedf(df_BA,df_ws_pdup)
df_Baw_wspdup.to_csv('IaI_Internetpdup.csv',index = False)
#
#
df_Baw_wspddown = mergedf(df_BA,df_ws_pddown)
df_Baw_wspddown.to_csv('IaI_Internetpddown.csv',index = False)
#
G_Baw = creat_G(df_Baw)
G_Baw_BApdup = creat_G(df_Baw_BApdup)
G_Baw_BApddown = creat_G(df_Baw_BApddown)
G_Baw_wspdup = creat_G(df_Baw_wspdup)
G_Baw_wspddown = creat_G(df_Baw_wspddown)


dp_Baw = nx.degree_assortativity_coefficient(G_Baw)
dp_Baw_BApdup = nx.degree_assortativity_coefficient(G_Baw_BApdup)
dp_Baw_BApddown = nx.degree_assortativity_coefficient(G_Baw_BApddown)
dp_Baw_wspdup = nx.degree_assortativity_coefficient(G_Baw_wspdup)
dp_Baw_wspddown = nx.degree_assortativity_coefficient(G_Baw_wspddown)



#df_BA = pd.read_csv('BA.csv')
#df_BA['network'] = 0
#df_ws = pd.read_csv('ws.csv')
#df_ws['network'] = 1
#df_BA_pdup = pd.read_csv('BA_pdup.csv')
#df_BA_pdup['network'] = 0
#df_ws_pdup = pd.read_csv('ws_pdup.csv')
#df_ws_pdup['network'] = 1
#df_BA_pddown = pd.read_csv('BA_pddown.csv')
#df_BA_pddown['network'] = 0
#df_ws_pddown = pd.read_csv('ws_pddown.csv')
#df_ws_pddown['network'] = 1
#df_Baw= pd.read_csv('Baw.csv')     
#df_Baw_BApddown = pd.read_csv('Baw_BApddown.csv')
#df_Baw_BApdup = pd.read_csv('Baw_BApdup.csv')
#df_Baw_wspddown = pd.read_csv('Baw_wspddown.csv')
#df_Baw_wspdup = pd.read_csv('Baw_wspdup.csv')            
#            
#G_BA = creat_G(df_BA)
#G_ws = creat_G(df_ws)
#G_BA_pdup = creat_G(df_BA_pdup)
#G_BA_pddown = creat_G(df_BA_pddown)
#G_ws_pdup = creat_G(df_ws_pdup)
#G_ws_pddown = creat_G(df_ws_pddown)
#G_Baw = creat_G(df_Baw)
#G_Baw_BApdup = creat_G(df_Baw_BApdup)
#G_Baw_BApddown = creat_G(df_Baw_BApddown)
#G_Baw_wspdup = creat_G(df_Baw_wspdup)
#G_Baw_wspddown = creat_G(df_Baw_wspddown)
##df_Baw = mergedf(df_BA,df_ws)
##df_Baw.to_csv('Baw.csv',index = False)
##
##
##df_Baw_BApdup = mergedf(df_BA_pdup,df_ws)
##df_Baw_BApdup.to_csv('Baw_BApdup.csv',index = False)
##
##
##df_Baw_BApddown = mergedf(df_BA_pddown,df_ws)
##df_Baw_BApddown.to_csv('Baw_BApddown.csv',index = False)
##
##
##df_Baw_wspdup = mergedf(df_BA,df_ws_pdup)
##df_Baw_wspdup.to_csv('Baw_wspdup.csv',index = False)
##
##
##df_Baw_wspddown = mergedf(df_BA,df_ws_pddown)
##df_Baw_wspddown.to_csv('Baw_wspddown.csv',index = False)
#
#dp_BA = nx.degree_assortativity_coefficient(G_BA)
#dp_BA_pdup = nx.degree_assortativity_coefficient(G_BA_pdup)
#dp_BA_pddown = nx.degree_assortativity_coefficient(G_BA_pddown)
#dp_ws = nx.degree_assortativity_coefficient(G_ws)
#dp_ws_pdup = nx.degree_assortativity_coefficient(G_ws_pdup)
#dp_ws_pddown = nx.degree_assortativity_coefficient(G_ws_pddown)
#dp_Baw = nx.degree_assortativity_coefficient(G_Baw)
#dp_Baw_BApdup = nx.degree_assortativity_coefficient(G_Baw_BApdup)
#dp_Baw_BApddown = nx.degree_assortativity_coefficient(G_Baw_BApddown)
#dp_Baw_wspdup = nx.degree_assortativity_coefficient(G_Baw_wspdup)
#dp_Baw_wspddown = nx.degree_assortativity_coefficient(G_Baw_wspddown)

