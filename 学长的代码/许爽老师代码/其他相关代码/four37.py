# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:56:54 2019

@author: PCC
"""

import pandas as pd
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
import pickle
import random
import copy
import nullmodel.tnm as nm

f = open('df_sd03_.pkl','rb')
df_ = pickle.load(f)
f.close()
G_all = nx.from_pandas_dataframe(df_,'n1','n2',edge_attr='timestamp',create_using=nx.MultiDiGraph()) 


'''
找到每个用户联系最密切的用户
'''
def closer(df_):
    df_['weight']=1
    list_more5node = list(set(df_['n1']).intersection(set(df_['n2'])))
    groupby_n1 = df_.groupby(['n1'])
    groupby_n2 = df_.groupby(['n2'])
    list_max = []
    for i in list_more5node:
        df_byn1 = groupby_n1.get_group(i)
        df_byn1 = df_byn1.drop(['timestamp','date'],axis=1)
        df_byn2 = groupby_n2.get_group(i)
        list_n1 = df_byn2['n1']
        list_n2 = df_byn2['n2']
        df_byn2_ = pd.DataFrame(({'n1':list_n2,'n2':list_n1,'weight':1}))
        a = df_byn1.append(df_byn2_)
        a = a.groupby(['n1', 'n2'])['weight'].sum().reset_index()
        a=a.sort_values(by="weight" , ascending=False)
        a = a.reset_index(drop = True)
        list_max.append([i,a['n2'][0]])
    return list_max

def Tt1(df_):
    list_more5node = list(set(df_['n1']).intersection(set(df_['n2'])))  
    groupby_n1 = df_.groupby(['n1'])
    groupby_n2 = df_.groupby(['n2'])
    list_Tt ,list_all_aveti,list_all_Ti= [],[],[]
    listd1 = []
    for i in list_more5node:
        df_byn1 =  groupby_n1.get_group(i)
        df_byn1 = df_byn1.reset_index(drop = True)
        df_byn2 =  groupby_n2.get_group(i)
        f = df_byn1.append(df_byn2)
        f=f.sort_values(by="timestamp" , ascending=True)
        f = f.reset_index(drop = True)
        list_Ti = []    #用户i接收消息然后给另一个用户发送短消息的所有间隔时间
        list_ti = []    #用户i发送两个连续消息的所有时间间隔
        listd2= []
        for k in range(len(df_byn1)-1):
            if df_byn1['n1'][k] == i and df_byn1['n1'][k+1] == i and df_byn1['timestamp'][k+1]-df_byn1['timestamp'][k] != 0:
                list_ti.append(df_byn1['timestamp'][k+1]-df_byn1['timestamp'][k])
        for j in range(len(f)-1):
            if f['n1'][j] != i and f['n1'][j+1] == i and f['timestamp'][j+1]-f['timestamp'][j] != 0:
                list_Ti.append(f['timestamp'][j+1]-f['timestamp'][j]) 
                listd2.append([list(f.iloc[j]),list(f.iloc[j+1])])
        if list_ti != []:
            list_all_Ti.append(list_Ti)    #用户i所有的Ti
            list_ti_ave = sum(list_ti)/len(list_ti)    #用户i的ti平均数
            list_all_aveti.append(list_ti_ave)     #所有用户的ti平均数
#            a = [k/list_ti_ave for k in list_Ti]
            a = []
            for k in range(len(list_Ti)):
                a.append(list_Ti[k]/list_ti_ave)   #用户i的所有Ti/<ti>
                if list_Ti[k]/list_ti_ave>1 and list_Ti[k]/list_ti_ave<1:
                    listd1.append([i,list_ti_ave,listd2[k]])
#                   print(i)
            list_Tt.extend(a)    #所有用户的所有Ti/<ti>
#    return list_Tt
    return list_Tt,list_all_aveti,list_all_Ti,listd1,listd2


def hist(list1):    
    list_bins = []
    for i in range(1001):
        list_bins.append(i*0.001)     
    for j in range(1,10):
        for k in range(1,1000):
            list_bins.append(j + k*0.001)
    list2 = []
    for i in list1:
        if i <10: 
            list2.append(i)
    xy = plt.hist(list2,bins = list_bins)
    plt.xscale('log')
    plt.yscale('log')
    return xy


def ave(list_x,list_y):
    list_x1 = []
    list_y1 = []
    for i in range(90):
        list_x1.append(list_x[99+i*10:109+i*10][0])
        list_y1.append(sum(list_y[99+i*10:109+i*10])/10)
    list_x2 = []
    list_y2 = []
    for i in range(90):
        list_x2.append(list_x[999+i*100:1099+i*100][0])
        list_y2.append(sum(list_y[999+i*100:1099+i*100])/100)        
    list_x_all = []
    list_x_all.extend(list_x[:99])
    list_x_all.extend(list_x1)
    list_x_all.extend(list_x2)
    list_y_all = []
    list_y_all.extend(list_y[:99])
    list_y_all.extend(list_y1)
    list_y_all.extend(list_y2)
    return list_x_all,list_y_all

def G2G2df(G): 
    dict_a = nx.get_edge_attributes(G,'timestamp')
    b = list(dict_a.keys())
    list_t = list(dict_a.values())
    list_n1,list_n2 = [],[]
    for i in b:
        list_n1.append(list(i)[0])
        list_n2.append(list(i)[1])
    df = pd.DataFrame(({'n1':list_n1,'n2':list_n2,'timestamp':list_t}))
    return df


def closer_part(df1):
    list_max = closer(df1)
    list1_,list2_= [],[]
    for i in range(len(df1)):
        if [df1['n1'][i], df1['n2'][i]] in list_max or [df1['n2'][i], df1['n1'][i]] in list_max:
            list1_.append(i)
        else:
            list2_.append(i)
    df_1 = df1.iloc[list1_]
    df_2 = df1.iloc[list2_]
    return df_1,df_2


def pdistribution3(xy_G2G2,xy_G2G2_all,xy_G2G2_pa):
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    list_x = list(xy_G2G2[1])[1:]
    list_y = [i/sum(list(xy_G2G2[0])) for i in list(xy_G2G2[0])]
    list_x_pa = list(xy_G2G2_pa[1])[1:]
    list_y_pa = [i/sum(list(xy_G2G2_pa[0])) for i in list(xy_G2G2_pa[0])]
    list_x_real = list(xy_real[1])[1:]
    list_y_real = [i/sum(list(xy_real[0])) for i in list(xy_real[0])]
    list_x_all = list(xy_G2G2_all[1])[1:]
    list_y_all = [i/sum(list(xy_G2G2_all[0])) for i in list(xy_G2G2_all[0])]
    list_x,list_y = ave(list_x,list_y)
    list_x_real,list_y_real = ave(list_x_real,list_y_real)
    list_x_all,list_y_all = ave(list_x_all,list_y_all)
    list_x_pa,list_y_pa = ave(list_x_pa,list_y_pa)
    plt.plot(list_x_real,list_y_real,'bo-',markersize =3, label='Real data')
    plt.plot(list_x_all,list_y_all,'ro-',markersize =3, label='All input shuffled')
#    plt.plot(list_x,list_y,'yo-',markersize =3, label='Secondary input shuffled')
    plt.plot(list_x_pa,list_y_pa,'yo-',markersize =3, label='Pricioal input shuffled')
    plt.legend(loc="best",fontsize=8)
    plt.xlabel('Ti/<ti>',fontsize=10)
    plt.ylabel('p(Ti/<ti>)',fontsize=10)
    plt.xticks(fontsize=10) 
    plt.yticks(fontsize=10)
    plt.xscale('log')
    plt.yscale('log')

def pdistribution4(xy_G2G2,xy_G2G2_all,xy_G2G2_pa,label='时变网络的接触置乱'):
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    list_x = list(xy_G2G2[1])[1:]
    list_y = [i/sum(list(xy_G2G2[0])) for i in list(xy_G2G2[0])]
    list_x_pa = list(xy_G2G2_pa[1])[1:]
    list_y_pa = [i/sum(list(xy_G2G2_pa[0])) for i in list(xy_G2G2_pa[0])]
    list_x_real = list(xy_real[1])[1:]
    list_y_real = [i/sum(list(xy_real[0])) for i in list(xy_real[0])]
    list_x_all = list(xy_G2G2_all[1])[1:]
    list_y_all = [i/sum(list(xy_G2G2_all[0])) for i in list(xy_G2G2_all[0])]
    list_x,list_y = ave(list_x,list_y)
    list_x_real,list_y_real = ave(list_x_real,list_y_real)
    list_x_all,list_y_all = ave(list_x_all,list_y_all)
    list_x_pa,list_y_pa = ave(list_x_pa,list_y_pa)
    plt.plot(list_x_real,list_y_real,'bo-',markersize =3, label='原始数据')
    plt.plot(list_x_all,list_y_all,'ro-',markersize =3, label='置乱全部用户')
    plt.plot(list_x,list_y,'o-',markersize =3, label='置乱非主要联系用户')
    plt.plot(list_x_pa,list_y_pa,'yo-',markersize =3, label='置乱主要联系用户')
    plt.legend(loc="best",fontsize=8)
    plt.xlabel('Ti/<ti>',fontsize=10)
    plt.ylabel('p(Ti/<ti>)',fontsize=10)
    plt.xticks(fontsize=10) 
    plt.yticks(fontsize=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('%s.png'%label, dpi=750)

if __name__=="__main__":
    '''
    原始数据
    '''
#    list1,list_all_aveti,list_all_Ti,listd1,listd2 = Tt1(df_)
#    xy_real = hist(list1)
#    
#    
#    '''
#    全部置乱
#    '''
#    
#    G3_all = nm.time_swap(G_all)
#    df_n3_all = G2G2df(G3_all)
#    list13_all,list_all_aveti3_all,list_all_Ti3_all,listd13_all,listd23_all = Tt1(df_n3_all)
#    xy_G2G3_all = hist(list13_all)
#    
#    
#    G7_all = nm.touch_swap(G_all)
#    df_n7_all = G2G2df(G7_all)
#    list17_all,list_all_aveti7_all,list_all_Ti7_all,listd17_all,listd27_all = Tt1(df_n7_all)
#    xy_G2G7_all = hist(list17_all)
#    
#    
#    
#    '''
#    置乱次要朋友
#    '''
#    df_1,df_2= closer_part(df_)
#    
#    G = nx.from_pandas_dataframe(df_2,'n1','n2',edge_attr='timestamp',create_using=nx.MultiDiGraph()) 
#    G3 = nm.time_swap(G)
#    G7 = nm.touch_swap(G)
#    
#    df_n3 = G2G2df(G3) 
#    df_n3 = df_n3.append(df_1)
#    df_n7 = G2G2df(G7)
#    df_n7 = df_n7 .append(df_1)
#    
#    list13,list_all_aveti3,list_all_Ti3,listd13,listd23 = Tt1(df_n3)
#    list17,list_all_aveti7,list_all_Ti7,listd17,listd27 = Tt1(df_n7)
#    
#    xy_G2G3 = hist(list13)
#    xy_G2G7 = hist(list17)
#    
#    '''
#    置乱主要朋友
#    '''
#    
#    G_1 = nx.from_pandas_dataframe(df_1,'n1','n2',edge_attr='timestamp',create_using=nx.MultiDiGraph()) 
#    
#    G3_pa = nm.time_swap(G_1)
#    df_n3_pa = G2G2df(G3_pa) 
#    df_n3_pa = df_n3_pa.append(df_2)
#    list13_pa,list_all_aveti3,list_all_Ti3,listd13,listd23 = Tt1(df_n3_pa)
#    xy_G2G3_pa = hist(list13_pa)
#    
#    G7_pa = nm.time_swap(G_1)
#    df_n7_pa = G2G2df(G7_pa)
#    df_n7_pa = df_n7_pa.append(df_2)
#    list17_pa,list_all_aveti7,list_all_Ti7,listd17,listd27 = Tt1(df_n7_pa)
#    xy_G2G7_pa = hist(list17_pa)
#    
#    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_real43.pkl','wb')
#    pickle.dump(xy_real,f)
#    f.close()
#    
#    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_G2G3_43.pkl','wb')
#    pickle.dump(xy_G2G3,f)
#    f.close()
#    
#    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_G2G3_all_43.pkl','wb')
#    pickle.dump(xy_G2G3_all,f)
#    f.close()
#    
#    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_G2G3_pa_43.pkl','wb')
#    pickle.dump(xy_G2G3_pa,f)
#    f.close()
#    
#    
#    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_G2G7_43.pkl','wb')
#    pickle.dump(xy_G2G7,f)
#    f.close()
#    
#    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_G2G7_all_43.pkl','wb')
#    pickle.dump(xy_G2G7_all,f)
#    f.close()
#    
#    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_G2G7_pa_43.pkl','wb')
#    pickle.dump(xy_G2G7_pa,f)
#    f.close()
    
    '''
    画图
    '''
    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_real43.pkl','rb')
    xy_real = pickle.load(f)
    f.close()
#    
    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_G2G3_43.pkl','rb')
    xy_G2G3 = pickle.load(f)
    f.close()
    
    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_G2G3_all_43.pkl','rb')
    xy_G2G3_all = pickle.load(f)
    f.close()

    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_G2G3_pa_43.pkl','rb')
    xy_G2G3_pa = pickle.load(f)
    f.close()
    
    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_G2G7_43.pkl','rb')
    xy_G2G7 = pickle.load(f)
    f.close()
    
    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_G2G7_all_43.pkl','rb')
    xy_G2G7_all = pickle.load(f)
    f.close()

    f = open('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\xy_G2G7_pa_43.pkl','rb')
    xy_G2G7_pa = pickle.load(f)
    f.close()
##
##
    pdistribution4(xy_G2G3,xy_G2G3_all,xy_G2G3_pa,label='时变网络的时间置乱算法')
    pdistribution4(xy_G2G7,xy_G2G7_all,xy_G2G7_pa,label='时变网络的接触置乱算法')
##    
##    
##    '''
##    第一个值
##    '''
    y = [i/sum(list(xy_real[0])) for i in list(xy_real[0])][0]
    
    y3_all = [i/sum(list(xy_G2G3_all[0])) for i in list(xy_G2G3_all[0])][0] #全部
    y3 = [i/sum(list(xy_G2G3[0])) for i in list(xy_G2G3[0])][0]             #次要
    y3_pa = [i/sum(list(xy_G2G3_pa[0])) for i in list(xy_G2G3_pa[0])][0]          #主要
##    
    y7_all = [i/sum(list(xy_G2G7_all[0])) for i in list(xy_G2G7_all[0])][0]
    y7 = [i/sum(list(xy_G2G7[0])) for i in list(xy_G2G7[0])][0]
    y7_pa = [i/sum(list(xy_G2G7_pa[0])) for i in list(xy_G2G7_pa[0])][0]






