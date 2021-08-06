# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 21:14:40 2020

@author: Administrator
"""

import pandas as pd
import networkx as nx


'''
把时间戳改为秒
'''
def txt2df_sd01(filename):
    df = pd.read_table(filename,sep = '\n',header =None)
    df[0][226397] = '   828   829 13190718'
    df[0][488541] = '  1001  1000 28001552'
    list1,list2,list3 = [],[],[]
    for i in range(len(df)):
        list1.append(int(df[0][i][:6]))
        list2.append(int(df[0][i][6:12]))
        list3.append(int(df[0][i][12:]))
    df1 = pd.DataFrame()
    df1['n1'] = list1
    df1['n2'] = list2
    df1['timestamp'] = list3
    df1['date'] = list3
    list_time = []
    for i in range(len(df1)):
        list_time.append(int(str(df1['timestamp'][i])[-2:]) + int(str(df1['timestamp'][i])[-4:-2])*60 + 
                          int(str(df1['timestamp'][i])[-6:-4])*3600 + int(str(df1['timestamp'][i])[-8:-6])*24*3600)
    df1['timestamp'] = list_time
    df1 = df1.reset_index(drop = True)
    return df1


def txt2df_sd03(filename):
    df = pd.read_table(filename,sep = '\n',header =None)
    
    df[0][95325] = '    18    17 21005226'
    list1,list2,list3 = [],[],[]
    for i in range(len(df)):
        list1.append(int(df[0][i][:6]))
        list2.append(int(df[0][i][6:12]))
        list3.append(int(df[0][i][12:]))
    df1 = pd.DataFrame()
    df1['n1'] = list1
    df1['n2'] = list2
    df1['timestamp'] = list3
    df1['date'] = list3
    list_time = []
    for i in range(len(df1)):
        list_time.append(int(str(df1['timestamp'][i])[-2:]) + int(str(df1['timestamp'][i])[-4:-2])*60 + 
                          int(str(df1['timestamp'][i])[-6:-4])*3600 + int(str(df1['timestamp'][i])[-8:-6])*24*3600)
    df1['timestamp'] = list_time
    df1 = df1.reset_index(drop = True)
    return df1



def txt2df_sd02(filename):
    df = pd.read_table(filename,sep = '\n',header =None)
    df[0][226396] = '  1112  1113 11215902'
    df[0][226397] = '    46  3399 11215902'
    df[0][357469] = '  1629  1630 17220705'
    list1,list2,list3 = [],[],[]
    for i in range(len(df)):
        print(i)
        list1.append(int(df[0][i][:6]))
        list2.append(int(df[0][i][6:12]))
        list3.append(int(df[0][i][12:]))
    df1 = pd.DataFrame()
    df1['n1'] = list1
    df1['n2'] = list2
    df1['timestamp'] = list3
    df1['date'] = list3
    list_time = []
    for i in range(len(df1)):
        list_time.append(int(str(df1['timestamp'][i])[-2:]) + int(str(df1['timestamp'][i])[-4:-2])*60 + 
                          int(str(df1['timestamp'][i])[-6:-4])*3600 + int(str(df1['timestamp'][i])[-8:-6])*24*3600)
    df1['timestamp'] = list_time
    df1 = df1.reset_index(drop = True)
    return df1
df_sd01 = txt2df_sd01('SD01.txt')
df_sd03 = txt2df_sd03('SD03.txt')
df_sd02 = txt2df_sd02('SD02.txt')
df_sms = pd.read_csv('E:/111111111111111111111111111111/new/论文/节点置换和信息传播对称性/sms_weight.csv')
df_call = pd.read_csv('E:/111111111111111111111111111111/new/论文/节点置换和信息传播对称性/call_weight.csv') 



G_sd01 = nx.from_pandas_dataframe(df,'n1','n2',edge_attr='timestamp',create_using=nx.MultiDiGraph()) 
G_sd02 = nx.from_pandas_dataframe(df_sd02,'n1','n2',edge_attr='timestamp',create_using=nx.MultiDiGraph()) 
G_sd03 = nx.from_pandas_dataframe(df_sd03,'n1','n2',edge_attr='timestamp',create_using=nx.MultiDiGraph()) 
G_sms = nx.from_pandas_dataframe(df_sms,'node1','node2',edge_attr='weight',create_using=nx.MultiDiGraph()) 
G_call = nx.from_pandas_dataframe(df_call,'node1','node2',edge_attr='weight',create_using=nx.MultiDiGraph()) 
'''
平均路径长度
'''
def shortest_path(G_call):
    pathlengths=[]
    #单源最短路径算法求出节点v到图G每个节点的最短路径，存入pathlengths
    
    for v in G_call.nodes():
        spl=nx.single_source_shortest_path_length(G_call,v)
    #    print('%s %s' % (v,spl))
        for p in spl.values():
            pathlengths.append(p)
    #取出每条路径，计算平均值。
#    print("average shortest path length %s" % (sum(pathlengths)/len(pathlengths)))
    return sum(pathlengths)/len(pathlengths)

sp1 = shortest_path(G_sd01)
sp2 = shortest_path(G_sd02)
sp3 = shortest_path(G_sd03)
sp_sms = shortest_path(G_sms)
sp_call = shortest_path(G_call)
'''
平均度
'''
def ave_degree(G):
#    all_degree = [i[1] for i in list(G.degree())]
    average_degree = sum(list(G.degree()))/len(list(G.degree()))
#    average_degree = sum(all_degree)/len(all_degree)
    return average_degree
ad1 = ave_degree(G_sd01)
ad2 = ave_degree(G_sd02)
ad3 = ave_degree(G_sd03)
'''
平均聚类系数
'''
def ave_clustering(G):
    return nx.average_clustering(G)
ac1 = ave_clustering(G_sd01)
ac2 = ave_clustering(G_sd02)
ac3 = ave_clustering(G_sd03)

'''
度匹配性
'''
def degree_as(G):
    return nx.degree_assortativity_coefficient(G)
da1 = degree_as(G_sd01)
da2 = degree_as(G_sd02)
da3 = degree_as(G_sd03)


def ave_strength(G_call):
    list_s = [G_call.degree(i,weight = 'weight') for i in list(G_call.nodes())]
    a = sum(list_s)/len(list_s)
    return a













