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





print(1)
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
#
G_call_pdup = unm.assort_mixing(G_call,nswap=2000000, max_tries=2000000, connected=1)
G_call_pddown = unm.disassort_mixing(G_call,nswap=2000000, max_tries=2000000, connected=1)
G_sms_pdup = unm.assort_mixing(G_sms,nswap=2000000, max_tries=2000000, connected=1)
G_sms_pddown = unm.disassort_mixing(G_sms,nswap=2000000, max_tries=2000000, connected=1)


dp_call = nx.degree_assortativity_coefficient(G_call)
dp_call_pdup = nx.degree_assortativity_coefficient(G_call_pdup)
dp_call_pddown = nx.degree_assortativity_coefficient(G_call_pddown)
dp_sms = nx.degree_assortativity_coefficient(G_sms)
dp_sms_pdup = nx.degree_assortativity_coefficient(G_sms_pdup)
dp_sms_pddown = nx.degree_assortativity_coefficient(G_sms_pddown)


'''
合并双层网络
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
df_cas.to_csv('cas.csv',index = False)
#
df_call_pdup = G2df(G_call_pdup)
df_call_pdup.to_csv('call_pdup.csv',index = False)
df_call_pdup['network'] = 0
df_cas_callpdup = mergedf(df_call_pdup,df_sms)
df_cas_callpdup.to_csv('cas_callpdup.csv',index = False)
###
df_call_pddown = G2df(G_call_pddown)
df_call_pddown.to_csv('call_pddown.csv',index = False)
df_call_pddown['network'] = 0
df_cas_callpddown = mergedf(df_call_pddown,df_sms)
df_cas_callpddown.to_csv('cas_callpddown.csv',index = False)
#
df_sms_pdup = G2df(G_sms_pdup)
df_sms_pdup.to_csv('sms_pdup.csv',index = False)
df_sms_pdup['network'] = 1
df_cas_smspdup = mergedf(df_sms_pdup,df_call)
df_cas_smspdup.to_csv('cas_smspdup.csv',index = False)
###
df_sms_pddown = G2df(G_sms_pddown)
df_sms_pddown.to_csv('sms_pddown.csv',index = False)
df_sms_pddown['network'] = 1
df_cas_smspddown = mergedf(df_sms_pddown,df_call)
df_cas_smspddown.to_csv('cas_smspddown.csv',index = False)



G_cas = creat_G(df_cas)
G_cas_callpdup = creat_G(df_cas_callpdup)
G_cas_callpddown = creat_G(df_cas_callpddown)
G_cas_smspdup = creat_G(df_cas_smspdup)
G_cas_smspddown = creat_G(df_cas_smspddown)


dp_cas = nx.degree_assortativity_coefficient(G_cas)
dp_cas_callpdup = nx.degree_assortativity_coefficient(G_cas_callpdup)
dp_cas_callpddown = nx.degree_assortativity_coefficient(G_cas_callpddown)
dp_cas_smspdup = nx.degree_assortativity_coefficient(G_cas_smspdup)
dp_cas_smspddown = nx.degree_assortativity_coefficient(G_cas_smspddown)


